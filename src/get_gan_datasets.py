import time
import torch

if __name__ == '__main__':
    while True:
        start = time.time()
        if total_pos_cnt >= total_neg_cnt:
            break
        else:
            # update the number of positive samples

            start1 = time.time()
            z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
            # z = vae.generate_z()
            end1 = time.time()
            print("Generate_z time : ", end1-start1)
            start2 = time.time()
            new_sample = gan.generate_samples(z)
            end2 = time.time()
            print("Generate_sample time : ", end2-start2)
            new_label = torch.tensor([1], device="cpu")

            target_dataset.samples = torch.cat(
                [
                    target_dataset.samples,
                    new_sample,
                ],
            )
            target_dataset.labels = torch.cat(
                [
                    target_dataset.labels,
                    new_label,
                ]
            )
            total_pos_cnt += 1

            # # update the number of overlapping positive samples
            # indices = get_knn_indices(new_sample, full_dataset.samples)
            # labels = full_dataset.labels[indices]

            # if 0 in labels:
            #     print("ol_pos_cnt int while : ", ol_pos_cnt)
            #     ol_pos_cnt += 1
            
        end = time.time()
        print("Generate sample time : ", end-start)

    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    print("target_dataset.samples : ", len(target_dataset.samples))
    print("target_dataset.labels : ", len(target_dataset.labels))