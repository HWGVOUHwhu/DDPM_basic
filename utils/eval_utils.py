from torch_fidelity import calculate_metrics


def compute_fid(real_dir, fake_dir):
    print(f"Calculating FID between {real_dir} and {fake_dir}...")
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=False
    )
    return metrics['frechet_inception_distance']
