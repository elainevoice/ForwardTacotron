import time
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_cos_matrix
from utils.dsp import reconstruct_waveform, np_now
from utils.files import unpickle_binary
from utils.paths import Paths
from utils.text import clean_text, text_to_sequence
from utils.text.cleaners import basic_cleaners


class ForwardTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: ForwardTacotron, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: ForwardTacotron,
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (s_id, semb, x, m, ids, x_lens, m_lens, dur) in enumerate(session.train_set, 1):

                start = time.time()
                model.train()
                x, m, dur, lens, s_id, semb = x.to(device), m.to(device), dur.to(device), m_lens.to(device), s_id.to(device), semb.to(device)

                m1_hat, m2_hat, dur_hat = model(x, m, dur, semb)

                m1_loss = self.l1_loss(m1_hat, m, lens)
                m2_loss = self.l1_loss(m2_hat, m, lens)

                dur_loss = F.l1_loss(dur_hat, dur)

                loss = m1_loss + m2_loss + dur_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                optimizer.step()
                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                if step % hp.forward_plot_every == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            m_val_loss, dur_val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', m_val_loss, model.get_step())
            self.writer.add_scalar('Duration_Loss/val', dur_val_loss, model.get_step())
            save_checkpoint('forward', self.paths, model, optimizer, is_silent=True)

            m_loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: ForwardTacotron, val_set: Dataset) -> Tuple[float, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, (s_id, semb, x, m, ids, x_lens, m_lens, dur) in enumerate(val_set, 1):
            s_id, semb, x, m, dur, m_lens = s_id.to(device), semb.to(device), x.to(device), m.to(device), dur.to(device), m_lens.to(device)
            with torch.no_grad():
                m1_hat, m2_hat, dur_hat = model(x, m, dur, semb)
                m1_loss = self.l1_loss(m1_hat, m, m_lens)
                m2_loss = self.l1_loss(m2_hat, m, m_lens)
                dur_loss = F.l1_loss(dur_hat, dur)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        return m_val_loss / len(val_set), dur_val_loss / len(val_set)

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        s_id, semb, x, m, ids, x_lens, m_lens, dur = session.val_sample
        x, semb, m, s_id, dur = x.to(device), semb.to(device), m.to(device), s_id.to(device), dur.to(device)

        # plot speaker cosine similarity matrix
        speaker_emb_dict = unpickle_binary(self.paths.data / 'speaker_emb_dict.pkl')
        speaker_token_dict = unpickle_binary(self.paths.data / 'speaker_token_dict.pkl')
        token_speaker_dict = {v: k for k, v in speaker_token_dict.items()}
        speaker_ids = sorted(list(speaker_emb_dict.keys()))[:20]
        embeddings = [speaker_emb_dict[s_id] for s_id in speaker_ids]
        cos_mat = cosine_similarity(embeddings)
        np.fill_diagonal(cos_mat, 0)
        cos_mat_fig = plot_cos_matrix(cos_mat, labels=speaker_ids)
        self.writer.add_figure('Embedding_Metrics/speaker_cosine_dist', cos_mat_fig, model.step)

        for idx in range(len(hp.val_speaker_ids)):
            x_len = x_lens[idx]
            m_len = m_lens[idx]
            m1_hat, m2_hat, dur_hat = model(x[idx:idx+1, :x_len], m[idx:idx+1, :, :m_len], dur[idx:idx+1], semb[idx:idx+1])
            m1_hat_np = np_now(m1_hat)
            m2_hat_np = np_now(m2_hat)
            gen_sid = int(s_id[idx].cpu())
            gen_semb = semb[idx].cpu()
            target_sid = token_speaker_dict[gen_sid]
            m1_hat = m1_hat_np[0, :, :]
            m2_hat = m2_hat_np[0, :, :]
            m_target_np = np_now(m)
            m_target = m_target_np[idx, :, :m_len]

            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)
            m_target_fig = plot_mel(m_target)

            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/target', m_target_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/postnet', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)
            target_wav = reconstruct_waveform(m_target)

            self.writer.add_audio(
                tag=f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Ground_Truth_Aligned_{idx}_SID_{target_sid}/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

            text = clean_text('Ideas engineering is changing the world for the better.')
            inputs = text_to_sequence(text)
            m1_hat, m2_hat, att = model.generate(inputs, gen_semb)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)
            m_target_fig = plot_mel(m_target)

            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/target', m_target_fig, model.step)
            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'Generated_{idx}_SID_{target_sid}/postnet', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag=f'Generated_{idx}_SID_{target_sid}/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Generated_{idx}_SID_{target_sid}/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)

            text = basic_cleaners('Eine gute Idee kann die Welt verändern.')
            inputs = text_to_sequence(text)
            m1_hat, m2_hat, att = model.generate(inputs, gen_semb)
            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)
            m_target_fig = plot_mel(m_target)

            self.writer.add_figure(f'Generated_DE_{idx}_SID_{target_sid}/target', m_target_fig, model.step)
            self.writer.add_figure(f'Generated_DE_{idx}_SID_{target_sid}/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'Generated_DE_{idx}_SID_{target_sid}/postnet', m2_hat_fig, model.step)

            m2_hat_wav = reconstruct_waveform(m2_hat)

            self.writer.add_audio(
                tag=f'Generated_DE_{idx}_SID_{target_sid}/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Generated_DE_{idx}_SID_{target_sid}/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=hp.sample_rate)
