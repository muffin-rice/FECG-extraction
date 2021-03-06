import os

import torch
from ecgdetectors import Detectors
detectors = Detectors(125)
detector = detectors.hamilton_detector

from vae import VAE
from load_data import ECGDataModule
from vae_train import DATA_DIR, NUM_DATA_WORKERS, BATCH_SIZE, NUM_TRAINER_WORKERS

def count_peak_matches(orig_signal, pred_signal, detector):
    orig_peaks = detector(orig_signal)
    pred_peaks = detector(pred_signal)

    ind = 0
    fp = 0
    tp = 0
    fn = 0

    for opeak in orig_peaks:
        omin = opeak - 7
        omax = opeak + 7
        while ind < len(pred_peaks):
            if pred_peaks[ind] >= omin and pred_peaks[ind] <= omax:
                ind += 1
                tp += 1
                break
            elif pred_peaks[ind] > omin:
                fn += 1
                break
            ind += 1
            fp += 1

    precision = tp / (tp + fp) if tp + fp else 1
    recall = tp / len(orig_peaks) if orig_peaks else int(bool(tp))

    return precision, recall, 2 * (precision * recall / (precision + recall) if precision + recall else 0)


run_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
model_name = 'modelv1.0'
model_version = 'version_23'
model_path = f'{run_root}/logging/{model_name}/{model_version}/checkpoints/last.ckpt'
output_root = f'{run_root}/output/{model_name}/{model_version}'

if __name__ == '__main__':
    model = VAE.load_from_checkpoint(model_path)
    model.eval()

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=1,
                       batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    model.eval()
    with torch.no_grad():
        psum = 0
        rsum = 0
        fsum = 0
        count = 0

        print('training scores:')
        dl = dm.train_dataloader()
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(), d['mecg_sig'][i][0].detach().cpu().numpy() + d['fecg_sig'][i][0].detach().cpu().numpy() - model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
                psum += p
                rsum += r
                fsum += f
                count += 1

        print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)

        print('testing scores:')
        psum = 0
        rsum = 0
        fsum = 0
        count = 0

        dl = dm.test_dataloader()
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(),
                                             d['mecg_sig'][i][0].detach().cpu().numpy() + d['fecg_sig'][i][
                                                 0].detach().cpu().numpy() - model_output['x_recon'][i][
                                                 0].detach().cpu().numpy(), detector)
                psum += p
                rsum += r
                fsum += f
                count += 1

        print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)

        print('validation scores:')
        psum = 0
        rsum = 0
        fsum = 0
        count = 0

        dl = dm.val_dataloader()
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(),
                                             d['mecg_sig'][i][0].detach().cpu().numpy() + d['fecg_sig'][i][
                                                 0].detach().cpu().numpy() - model_output['x_recon'][i][
                                                 0].detach().cpu().numpy(), detector)
                psum += p
                rsum += r
                fsum += f
                count += 1

        print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)
