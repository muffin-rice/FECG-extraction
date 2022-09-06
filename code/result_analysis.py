import os

import numpy as np
import torch
from ecgdetectors import Detectors
detectors = Detectors(125)
detector = detectors.hamilton_detector

peak_detectors = [detectors.hamilton_detector, detectors.christov_detector, detectors.engzee_detector, detectors.pan_tompkins_detector, detectors.swt_detector, detectors.two_average_detector]



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
model_name = 'modelv1.2'
model_version = 'version_0'
model_path = f'{run_root}/logging/{model_name}/{model_version}/checkpoints/last.ckpt'
output_root = f'{run_root}/output/{model_name}/{model_version}'

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = VAE.load_from_checkpoint(model_path)

    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=1,
                       batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    model.eval()
    with torch.no_grad():
        # psum = np.zeros(len(peak_detectors))
        # rsum = np.zeros(len(peak_detectors))
        # fsum = np.zeros(len(peak_detectors))
        # count = 0
        #
        # print('training scores:')
        # dl = dm.train_dataloader()
        # for j, d in enumerate(dl):
        #     model_output = model(d)
        #
        #     for i in range(d['fecg_sig'].shape[0]):
        #         if d['snr'][i].detach().cpu().numpy():
        #             p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(), model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
        #             plt.plot(model_output['x_recon'][i][0].detach().cpu().numpy())
        #             plt.plot(d['fecg_sig'][i][0].detach().cpu().numpy())
        #             plt.show()
    #                 # for j in range(len(peak_detectors)):
    #                 #     p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(), model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
    #                 #     plt.plot(d['mecg_sig'][i][0].detach().cpu().numpy() + d['fecg_sig'][i][0].detach().cpu().numpy() - model_output['x_recon'][i][0].detach().cpu().numpy())
    #                 #     plt.plot(d['fecg_sig'][i][0].detach().cpu().numpy())
    #                 #     plt.show()
    #                 #     # exit()
    #                 #     psum[j] += p
    #                 #     rsum[j] += r
    #                 #     fsum[j] += f
    #                 #     count += 1
    #
    #     print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)
    #
        print('testing scores:')
        psum = np.zeros(len(peak_detectors))
        rsum = np.zeros(len(peak_detectors))
        fsum = np.zeros(len(peak_detectors))
        count = 0

        dl = dm.test_dataloader()
        for j, d in enumerate(dl):
            from scipy.signal import savgol_filter, resample
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                # points = d['mecg_sig'][i][0]
                # filtered = savgol_filter(points, window_length=10, polyorder=6)
                # plt.plot(points, color='b')
                # plt.plot(filtered, color='r')
                # plt.show()
                if d['snr'][i].detach().cpu().numpy():
                    p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(), model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.set_title("prediction")
                    ax1.plot(model_output['x_recon'][i][0].detach().cpu().numpy())
                    ax2.set_title("ground truth")
                    ax2.plot(d['fecg_sig'][i][0].detach().cpu().numpy())
                    ax3.set_title("mecg")
                    ax3.plot(d['mecg_sig'][i][0].detach().cpu().numpy())
                    plt.show()
                    # raise SystemExit
                    # for j in range(len(peak_detectors)):
                    #     p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(),
                    #                                  model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
                    #     plt.plot(d['mecg_sig'][i][0].detach().cpu().numpy() + d['fecg_sig'][i][0].detach().cpu().numpy() - model_output['x_recon'][i][0].detach().cpu().numpy())
                    #     plt.plot(d['fecg_sig'][i][0].detach().cpu().numpy())
                    #     plt.show()
                    #     # exit()
                    #     psum[j] += p
                    #     rsum[j] += r
                    #     fsum[j] += f
                    #     count += 1

        print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)

        print('validation scores:')
        psum = np.zeros(len(peak_detectors))
        rsum = np.zeros(len(peak_detectors))
        fsum = np.zeros(len(peak_detectors))
        count = 0

        dl = dm.val_dataloader()
        for j, d in enumerate(dl):
            model_output = model(d)

            for i in range(d['fecg_sig'].shape[0]):
                if d['snr'][i].detach().cpu().numpy():
                    p, r, f = count_peak_matches(d['fecg_sig'][i][0].detach().cpu().numpy(), model_output['x_recon'][i][0].detach().cpu().numpy(), detector)
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
                    ax1.set_title("prediction")
                    ax1.plot(model_output['x_recon'][i][0].detach().cpu().numpy())
                    ax2.set_title("ground truth")
                    ax2.plot(d['fecg_sig'][i][0].detach().cpu().numpy())
                    ax3.set_title("mecg")
                    ax3.plot(d['mecg_sig'][i][0].detach().cpu().numpy())
                    plt.show()
                    # exit()
                    # for j in range(len(peak_detectors)):
                    #     psum[j] += p
                    #     rsum[j] += r
                    #     fsum[j] += f
                    #     count += 1

        print('precision', psum / count, 'recall', rsum / count, 'f1', fsum / count)
