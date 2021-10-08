import glob
import tqdm
import numpy as np
import h5py
import SimpleITK as sitk


def resample_to_spacing(image, new_spacing):
    spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                zip(original_size, spacing, new_spacing)]
    img_dcm = sitk.Resample(image, new_size, sitk.Transform(),
                            sitk.sitkNearestNeighbor, image.GetOrigin(), new_spacing,
                            image.GetDirection(), 0, image.GetPixelID())
    return img_dcm


def convert_to_3D(directory, label):
    images = []
    files = glob.glob(f"{directory}/*.dcm")
    for file in files:
        img_dcm = sitk.ReadImage(file)

        # resample to isotropic spacing if needed
        spacing = img_dcm.GetSpacing()
        new_spacing = (1.0, 1.0, 1.0)
        if spacing != new_spacing:
            img_dcm = resample_to_spacing(img_dcm, new_spacing)
        img_arr = sitk.GetArrayFromImage(img_dcm)
        images.append(np.transpose(img_arr, (2, 1, 0))[:, :, 0])

    # create 3D array
    img_shape = list(images[0].shape)
    img_shape.append(len(images))
    img_3D = np.zeros(img_shape)

    # fill in 3D array with images from the files
    for idx, s in enumerate(images):
        img_3D[:, :, idx] = s

    # resample image so that it matches size of the mask
    img_img = sitk.GetImageFromArray(np.transpose(img_3D, (2, 1, 0)))
    label_img = sitk.GetImageFromArray(np.transpose(label, (2, 1, 0)))
    matched_img = sitk.Resample(img_img, label_img.GetSize(), sitk.Transform(),
                                sitk.sitkLinear, img_img.GetOrigin(),
                                img_img.GetSpacing(), img_img.GetDirection(),
                                0, img_img.GetPixelID())
    return np.transpose(sitk.GetArrayFromImage(matched_img), (2, 1, 0))


def get_label_data(directory):
    label_path = glob.glob(f"{directory}/*.gz")[0]
    label = sitk.ReadImage(label_path)
    spacing = label.GetSpacing()
    new_spacing = (1.0, 1.0, 1.0)
    if spacing != new_spacing:
        label = resample_to_spacing(label, new_spacing)
    label_array = sitk.GetArrayFromImage(label)
    return np.transpose(label_array, (2, 1, 0))


def process_dataset(image, label, directory):
    output_size = [96, 96, 96]
    label = (label == 1).astype(np.uint8)
    w, h, d = label.shape

    # center image around ground truth
    tempL = np.nonzero(label)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    px = max(output_size[0] - (maxx - minx), 0) // 2
    py = max(output_size[1] - (maxy - miny), 0) // 2
    pz = max(output_size[2] - (maxz - minz), 0) // 2
    minx = max(minx - np.random.randint(10, 20) - px, 0)
    maxx = min(maxx + np.random.randint(10, 20) + px, w)
    miny = max(miny - np.random.randint(10, 20) - py, 0)
    maxy = min(maxy + np.random.randint(10, 20) + py, h)
    minz = max(minz - np.random.randint(5, 10) - pz, 0)
    maxz = min(maxz + np.random.randint(5, 10) + pz, d)

    # normalize to zero mean and unit std dev.
    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)
    image = image[minx:maxx, miny:maxy]
    label = label[minx:maxx, miny:maxy]
    print(image.shape)
    print(label.shape)
    sitk.WriteImage(sitk.GetImageFromArray(image), f"{directory}/image.nrrd")
    sitk.WriteImage(sitk.GetImageFromArray(label), f"{directory}/label.nrrd")
    # f = h5py.File(f"{directory}/ct_norm2.h5", 'w')
    # f.create_dataset('image', data=image, compression="gzip")
    # f.create_dataset('seg', data=label, compression="gzip")
    # f.close()


if __name__ == "__main__":
    img_dirs = glob.glob("../data/Pancreas_CT/*/")
    for directory in tqdm.tqdm(img_dirs):
        label = get_label_data(directory)
        image = convert_to_3D(directory, label)
        process_dataset(image, label, directory)
        break  # testing
