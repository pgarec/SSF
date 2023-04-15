import numpy as np

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    rotated_feat = np.einsum('ti,tij->tj', features, rotations)
    data_plus_labels = np.concatenate((rotated_feat, labels.reshape(-1,1)),axis=1)

    permuted_dataset = np.random.permutation(data_plus_labels)

    data = permuted_dataset[:,0:2]
    shuffled_labels = permuted_dataset[:,2:]
    # return 10*np.random.permutation(np.einsum('ti,tij->tj', features, rotations))
    return data, shuffled_labels