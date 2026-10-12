import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

from torch.utils.data import Dataset

from .augmentation import SeismicAugmentation

class SeismicDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', num_rets=2, local_norm = True):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.num_rets = num_rets
        self.file = None
        self.split = split

        # Open once to collect info
        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            if split == 'train':
                self.dataset_names = list(f.keys())[:-2]
            elif split == 'test':
                self.dataset_names = list(f.keys())[-2:]
            elif split == 'seap':
                self.dataset_names = [k for k in list(f.keys()) if 'seap' in k]
            else:
                self.dataset_names = list(f.keys())
            self.length = len(f[self.dataset_names[0]])
            for name in self.dataset_names:
                self.data[name] = f[name][:]

            # Precompute mean/std for each dataset
            self.stats = {
                name: (float(np.mean(f[name][:])), float(np.std(f[name][:])))
                for name in self.dataset_names
            }
            self.resolutions = None
            self.res_counter = 0
        self.local_norm = local_norm


    def generate_batch(self, num_positions):
    
        sequence = []
        res_options = len(self.dataset_names)
        for i in range(num_positions):
            candidates = list(range(res_options))
            if i > 0:
                # Avoid repeating previous
                candidates.remove(sequence[-1])
            chosen = random.choice(candidates)
            sequence.append(chosen)
    
        # Ensure first != last
        if sequence[0] == sequence[-1]:
            candidates = list(range(res_options))
            candidates.remove(sequence[-2])  # Avoid matching previous
            if sequence[0] in candidates:
                candidates.remove(sequence[0])  # Avoid matching first
            sequence[-1] = random.choice(candidates)
    
        self.resolutions = sequence
        self.res_counter = 0

    def get_sorted_idx(self):
        return np.argsort(self.resolutions)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        # Choose random datasets
        if self.resolutions is None:
            chosen_dsets = random.sample(self.dataset_names, self.num_rets)
        else:
            chosen_idx = [self.resolutions[self.res_counter - i] for i in range(self.num_rets)]
            chosen_dsets = [self.dataset_names[i] for i in chosen_idx]
            self.res_counter += 1
        results = []

        for name in chosen_dsets:
            if self.local_norm:
                sample = self.data[name][idx].astype(np.float32)
                mean = np.mean(sample)
                std = np.std(sample)
                normalized = (sample - mean) / (std + 1e-8)
                results.append(torch.from_numpy(normalized).unsqueeze(0))  # shape: [1, H, W]
            else:
                sample = self.data[name][idx].astype(np.float32)
                mean, std = self.stats[name]
                normalized = (sample - mean) / (std + 1e-8)
                results.append(torch.from_numpy(normalized).unsqueeze(0))  # shape: [1, H, W]
        if len(results) == 1:
            return results[0]
        return results  # list of Tensors: [num_rets x 1 x H x W]

class UnifiedDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', local_norm = True, do_augment=True):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split

        # Open once to collect info
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            if split == 'train':
                self.model_names = list(f.keys())[:-2]
            elif split == 'test':
                self.model_names = list(f.keys())[-2:]
            else:
                self.model_names = list(f.keys())
                
            self.dataset_names = []
            for model_name in self.model_names:
                k = list(f[model_name].keys())
                for kk in k:
                    self.dataset_names.append(f'{model_name}/{kk}')
                
            # Precompute mean/std for each dataset
            self.stats = {
                name: (float(np.mean(f[name][:])), float(np.std(f[name][:])))
                for name in self.dataset_names
            }
            
            for name in self.dataset_names:
                if not local_norm:
                    mean, std = self.stats[name]
                    normalized = (f[name][:] - mean) / (std + 1e-8)
                    for s in normalized:
                        self.data.append(s[None])
                else:
                    base = f[name][:]
                    for s in base:
                        mean = np.mean(s)
                        std = np.std(s)
                        normalized = (s - mean) / (std + 1e-8)
                        self.data.append(normalized[None])

        print(f"Initialized dataset with {len(self.data)} images")
        self.do_augment = do_augment
        self.augment = SeismicAugmentation()

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.do_augment:
            return self.augment(torch.tensor(self.data[idx]), per_image=False)
        return self.data[idx]

class TripleDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', local_norm = True, do_augment=True):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split

        # Open once to collect info
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            if split == 'train':
                self.model_names = list(f.keys())[:-2]
            elif split == 'test':
                self.model_names = list(f.keys())[-2:]
            else:
                self.model_names = list(f.keys())
                
            self.dataset_names = []
            for model_name in self.model_names:
                k = list(f[model_name].keys())
                for kk in k:
                    self.dataset_names.append(f'{model_name}/{kk}')
                
            # Precompute mean/std for each dataset
            self.stats = {
                name: (float(np.mean(f[name][:])), float(np.std(f[name][:])))
                for name in self.dataset_names
            }
            
            for name in self.dataset_names:
                if not local_norm:
                    mean, std = self.stats[name]
                    normalized = (f[name][:] - mean) / (std + 1e-8)
                    for s in normalized:
                        self.data.append(s[None])
                else:
                    base = f[name][:]
                    for s in base:
                        mean = np.mean(s)
                        std = np.std(s)
                        if std > 0.05:
                            normalized = (s - mean) / (std + 1e-8)
                            self.data.append(normalized[None])

        print(f"Initialized dataset with {len(self.data)} images")
        self.do_augment = do_augment
        self.augment = SeismicAugmentation()

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Always include the requested idx image as the anchor
        base = torch.tensor(self.data[idx])
        if self.do_augment:
            base = self.augment(base, per_image=False)
    
        # Sample two extra random indices
        idx_p = torch.randint(0, len(self.data), (1,)).item()
        idx_n = torch.randint(0, len(self.data), (1,)).item()
     
        pos = torch.tensor(self.data[idx_p])
        neg = torch.tensor(self.data[idx_n])
    
        if self.do_augment:
            pos = self.augment(pos, per_image=False)
            neg = self.augment(neg, per_image=False)
    
        return base, pos, neg


class CondeDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', local_norm=True, emb_model=None):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split
        self.embs=None
            

        self.data = {}
        self.length = 0
        with h5py.File(h5_path, 'r') as f:
            if split == 'train':
                self.model_names = [n for n in list(f.keys()) if 'Modelo6' not in n]
            elif split == 'test':
                self.model_names = [n for n in list(f.keys()) if 'Modelo6' in n]
            elif split == 'm3':
                self.model_names = [n for n in list(f.keys()) if 'Modelo3' in n]
            else:
                self.model_names = list(f.keys())
            print(self.model_names)
            self.dataset_names = []
            for model_name in self.model_names:
                k = list(f[model_name].keys())
                for kk in k:
                    self.dataset_names.append(f'{model_name}/{kk}')

            if local_norm:
                for name in self.dataset_names:
                    data = np.array(f[name][:])
                    z_scores=[]
                    for i, s in enumerate(data):
                        mean = np.mean(s)
                        std = np.std(s)

                        z_scores.append((s - mean) / (std + 1e-6))
                            
                    self.data[name] = np.array(z_scores)
                    self.length += len(self.data[name])
            else:
                for name in self.dataset_names:
                    data = np.array(f[name][:])
                    mean = np.mean(data, axis=1, keepdims=True)
                    std = np.std(data, axis=1, keepdims=True)
    
                    z_scores = (data - mean) / (std + 1e-6)
                    self.data[name] = z_scores
    
                    self.length += len(self.data[name])
                    
            if emb_model is not None:
                self.embs = {}
                with torch.no_grad():
                    B = 1024
                    for name in tqdm(self.dataset_names, desc="Extracting Embs..."):
                        data = self.data[name]
                        embs = []
                        for i in range(0, len(data), B):
                            cube = data[i:i+B]
                            cube = torch.tensor(cube).unsqueeze(1).cuda()
                            embs.append(emb_model.encode(cube).cpu())
                
                        self.embs[name] = torch.cat(embs, dim=0)

        self.augment = SeismicAugmentation()
        print(f"Finished loading dataset with {self.length} images")
        


    def __len__(self):
        return self.length

    def get_idx(self, idx):
        for key in self.data.keys():
            n = len(self.data[key])
            if n <= idx:
                idx -= n
            else:
                return self.data[key][idx], key, idx
    def get_random_from_res(self, res, exclude=None):
        n = len(self.data[res])
        seismic = None
        while True:
            idx = random.randint(0, n-1)
            
            if exclude is None or not exclude==idx:
                data = self.data[res][idx]
                return data, idx

    def get_matching_structure(self, r1, s1_idx):
        ref = r1.split('/')[0]
        matching_res = []
        for key in self.data.keys():
            model = key.split('/')[0]
            if model==ref and not key == r1:
                matching_res.append(key)
        n = len(matching_res)
        if n==1:
            i=0
        else:
            i = random.randint(0, n-1)
        return self.data[matching_res[i]][s1_idx], matching_res[i]

    def __getitem__(self, idx):
        
        s1r1, r1, s1_idx = self.get_idx(idx)
        s2r1, s2_idx = self.get_random_from_res(r1, exclude=s1_idx)
        s1r2, r2 = self.get_matching_structure(r1, s1_idx)
        
        if self.embs is None:
            return torch.tensor(s1r1).unsqueeze(0), torch.tensor(s2r1).unsqueeze(0), torch.tensor(s1r2).unsqueeze(0)
        else:
            return torch.tensor(s1r1).unsqueeze(0), torch.tensor(s2r1).unsqueeze(0), torch.tensor(s1r2).unsqueeze(0), self.embs[r1][s1_idx], self.embs[r1][s2_idx], self.embs[r2][s1_idx]

class ResidualCondeDataset(Dataset):
    def __init__(self, h5_path=None, split="train", local_norm=True, model_names=None, dataset_names=None, return_rgt=True, emb_model=None, emb_batch_size=1024, embedding_device=None, reflectivity_token="Refletividade", emb_input_size=128):
        super().__init__()

        assert h5_path is not None, "You must provide a path to the h5 dataset"

        self.h5_path = h5_path
        self.split = split
        self.local_norm = local_norm
        self.return_rgt = return_rgt
        self.reflectivity_token = reflectivity_token.lower()
        # Spatial size the embedding model was built for (stored slices are larger, e.g. 224x224).
        self.emb_input_size = (emb_input_size, emb_input_size) if isinstance(emb_input_size, int) else tuple(emb_input_size)

        self.data = {}
        self.rgts = {}
        self.embs = None
        self.samples = []

        with h5py.File(h5_path, "r") as f:
            all_model_names = list(f.keys())

            if dataset_names is None:
                self.model_names = self._select_model_names(all_model_names, split, model_names)
                self.dataset_names = self._find_seismic_datasets(f, self.model_names)
            else:
                self.dataset_names = list(dataset_names)
                self.model_names = sorted({name.split("/", 1)[0] for name in self.dataset_names})

                if model_names is not None:
                    requested_models = set(model_names)
                    invalid_models = set(self.model_names) - requested_models

                    if invalid_models:
                        raise ValueError(f"dataset_names contains models not present in model_names: {sorted(invalid_models)}")

            self._validate_dataset_paths(f)

            print("Model names:", self.model_names)
            print("Dataset names:", self.dataset_names)

            model_rgts = self._load_model_rgts(f)

            for seismic_path in self.dataset_names:
                seismic = np.asarray(f[seismic_path][:], dtype=np.float32)
                seismic = np.nan_to_num(seismic, nan=0.0, posinf=0.0, neginf=0.0)
                seismic = self._normalize_seismic(seismic)

                if len(seismic) < 2:
                    raise ValueError(f"Dataset {seismic_path} contains only {len(seismic)} sample(s). At least two are required to select a positive sample.")

                self.data[seismic_path] = seismic

                model_name = seismic_path.split("/", 1)[0]
                self.rgts[seismic_path] = model_rgts.get(model_name)

                for local_idx in range(len(seismic)):
                    self.samples.append((seismic_path, local_idx))

        self._validate_negative_sampling()

        if emb_model is not None:
            self._extract_embeddings(emb_model, emb_batch_size, embedding_device)

        print(f"Finished loading dataset with {len(self.samples)} images")

    def _select_model_names(self, all_model_names, split, model_names):
        if model_names is not None:
            selected_models = list(model_names)
        elif isinstance(split, (list, tuple, set)):
            selected_models = list(split)
        elif split == "train":
            selected_models = [name for name in all_model_names if name.lower() != "buzios"]
        elif split == "buzios":
            selected_models = [name for name in all_model_names if name.lower() == "buzios"]
        elif split == "all":
            selected_models = list(all_model_names)
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'buzios', 'all', or a list of model names.")

        missing_models = [name for name in selected_models if name not in all_model_names]

        if missing_models:
            raise ValueError(f"Models not found in HDF5 file: {missing_models}")

        if not selected_models:
            raise ValueError("No models were selected.")

        return selected_models

    def _find_seismic_datasets(self, h5_file, model_names):
        dataset_names = []

        for model_name in model_names:
            for key in h5_file[model_name].keys():
                if "rgt" not in key.lower():
                    dataset_names.append(f"{model_name}/{key}")

        if not dataset_names:
            raise ValueError("No seismic datasets were found for the selected models.")

        return dataset_names

    def _validate_dataset_paths(self, h5_file):
        for dataset_name in self.dataset_names:
            if "/" not in dataset_name:
                raise ValueError(f"Invalid dataset path '{dataset_name}'. Expected a path such as 'model/dataset'.")

            if dataset_name not in h5_file:
                raise ValueError(f"Dataset '{dataset_name}' was not found in the HDF5 file.")

            dataset_key = dataset_name.split("/", 1)[1]

            if "rgt" in dataset_key.lower():
                raise ValueError(f"RGT dataset '{dataset_name}' cannot be used as a seismic dataset.")

    def _load_model_rgts(self, h5_file):
        model_rgts = {}

        if not self.return_rgt:
            return model_rgts

        for model_name in self.model_names:
            keys = list(h5_file[model_name].keys())
            rgt_keys = sorted([key for key in keys if "rgt" in key.lower()])

            if not rgt_keys:
                model_rgts[model_name] = None
                continue

            if len(rgt_keys) > 1:
                print(f"Warning: multiple RGT datasets found for {model_name}: {rgt_keys}. Using {rgt_keys[0]}.")

            rgt_path = f"{model_name}/{rgt_keys[0]}"
            model_rgts[model_name] = np.asarray(h5_file[rgt_path][:], dtype=np.float32)

        return model_rgts

    def _normalize_seismic(self, seismic):
        if seismic.ndim < 2:
            raise ValueError(f"Expected seismic data with at least two dimensions, but received shape {seismic.shape}.")

        if self.local_norm:
            normalization_axes = tuple(range(1, seismic.ndim))
            mean = np.mean(seismic, axis=normalization_axes, keepdims=True)
            std = np.std(seismic, axis=normalization_axes, keepdims=True)
        else:
            mean = np.mean(seismic, axis=1, keepdims=True)
            std = np.std(seismic, axis=1, keepdims=True)

        return ((seismic - mean) / (std + 1e-6)).astype(np.float32)

    def _validate_negative_sampling(self):
        if len(self.dataset_names) < 2:
            raise ValueError("At least two seismic datasets are required to create negative samples.")

        for seismic_path in self.dataset_names:
            exclude_reflectivity = self.reflectivity_token in seismic_path.lower()
            candidates = self._get_negative_candidates(seismic_path, exclude_reflectivity)

            if not candidates:
                raise ValueError(f"No valid negative dataset is available for anchor dataset '{seismic_path}'.")

    def _get_negative_candidates(self, exclude=None, exclude_reflectivity=False):
        candidates = [name for name in self.dataset_names if name != exclude]

        if exclude_reflectivity:
            candidates = [name for name in candidates if self.reflectivity_token not in name.lower()]

        return candidates

    def _extract_embeddings(self, emb_model, emb_batch_size, embedding_device):
        self.embs = {}

        if embedding_device is None:
            try:
                device = next(emb_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device(embedding_device)
            emb_model.to(device)

        was_training = getattr(emb_model, "training", False)
        emb_model.eval()

        with torch.no_grad():
            for dataset_name in self.dataset_names:
                data = self.data[dataset_name]
                embedding_batches = []

                for start_idx in range(0, len(data), emb_batch_size):
                    seismic_batch = torch.from_numpy(data[start_idx:start_idx + emb_batch_size]).float().unsqueeze(1).to(device)
                    seismic_batch = self._center_crop(seismic_batch, self.emb_input_size)
                    embeddings = emb_model.encode(seismic_batch)

                    if not torch.is_tensor(embeddings):
                        raise TypeError("emb_model.encode(...) must return a torch.Tensor.")

                    embedding_batches.append(embeddings.detach().cpu())

                self.embs[dataset_name] = torch.cat(embedding_batches, dim=0)

        if was_training:
            emb_model.train()

    @staticmethod
    def _center_crop(batch, size):
        out_h, out_w = size
        h, w = batch.shape[-2:]

        if h < out_h or w < out_w:
            raise ValueError(f"Input {h}x{w} is smaller than embedding input size {out_h}x{out_w}.")

        top = (h - out_h) // 2
        left = (w - out_w) // 2
        return batch[..., top:top + out_h, left:left + out_w]

    def _get_anchor_rgt(self, seismic_path, local_idx, anchor):
        rgt = self.rgts.get(seismic_path)

        if rgt is None:
            return torch.full_like(anchor, -1.0)

        if rgt.ndim >= 3 and rgt.shape[0] == len(self.data[seismic_path]):
            rgt = rgt[local_idx]

        rgt = torch.as_tensor(rgt, dtype=torch.float32)

        if rgt.ndim == anchor.ndim - 1:
            rgt = rgt.unsqueeze(0)

        finite_mask = torch.isfinite(rgt)

        if not finite_mask.any():
            return torch.full_like(rgt, -1.0)

        normalized_rgt = torch.full_like(rgt, -1.0)
        valid_values = rgt[finite_mask]
        minimum = valid_values.min()
        maximum = valid_values.max()
        normalized_rgt[finite_mask] = (valid_values - minimum) / (maximum - minimum + 1e-6)

        return normalized_rgt

    def __len__(self):
        return len(self.samples)

    def get_idx(self, idx):
        seismic_path, local_idx = self.samples[idx]
        return self.data[seismic_path][local_idx], seismic_path, local_idx

    def get_random_res(self, exclude=None, exclude_reflectivity=False):
        candidates = self._get_negative_candidates(exclude, exclude_reflectivity)

        if not candidates:
            raise RuntimeError(f"No negative dataset is available after excluding '{exclude}'.")

        return random.choice(candidates)

    def get_random_from_res(self, seismic_path, exclude=None):
        dataset_length = len(self.data[seismic_path])

        if exclude is not None and dataset_length < 2:
            raise RuntimeError(f"Cannot exclude index {exclude} from dataset '{seismic_path}' because it contains fewer than two samples.")

        while True:
            local_idx = random.randrange(dataset_length)

            if exclude is None or local_idx != exclude:
                return self.data[seismic_path][local_idx], local_idx

    def __getitem__(self, idx):
        anchor_array, anchor_path, anchor_idx = self.get_idx(idx)
        positive_array, positive_idx = self.get_random_from_res(anchor_path, exclude=anchor_idx)

        anchor_is_reflectivity = self.reflectivity_token in anchor_path.lower()
        negative_path = self.get_random_res(exclude=anchor_path, exclude_reflectivity=anchor_is_reflectivity)
        negative_array, negative_idx = self.get_random_from_res(negative_path)

        anchor = torch.from_numpy(anchor_array).float().unsqueeze(0)
        positive = torch.from_numpy(positive_array).float().unsqueeze(0)
        negative = torch.from_numpy(negative_array).float().unsqueeze(0)

        outputs = [anchor, positive, negative]

        if self.return_rgt:
            outputs.append(self._get_anchor_rgt(anchor_path, anchor_idx, anchor))

        if self.embs is not None:
            outputs.extend([self.embs[anchor_path][anchor_idx], self.embs[anchor_path][positive_idx], self.embs[negative_path][negative_idx]])

        return tuple(outputs)


class ResidualDeepfault(Dataset):
    def __init__(self, h5_path, split="train", local_norm=True, preload=True):
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'.")

        self.h5_path = h5_path
        self.split = split
        self.local_norm = local_norm
        self.preload = preload
        self.dataset_path = f"df/{split}_sis"
        self.data = None

        with h5py.File(self.h5_path, "r") as h5_file:
            if "df" not in h5_file:
                raise KeyError("The HDF5 file does not contain the 'df' group.")

            if self.dataset_path not in h5_file:
                available = list(h5_file["df"].keys())
                raise KeyError(f"Dataset '{self.dataset_path}' was not found. Available datasets under 'df': {available}")

            self.num_images = len(h5_file[self.dataset_path])

            if self.num_images < 2:
                raise ValueError(f"Dataset '{self.dataset_path}' must contain at least two images, but contains {self.num_images}.")

            self.image_shape = h5_file[self.dataset_path].shape[1:]

            if self.preload:
                self.data = np.asarray(h5_file[self.dataset_path][:], dtype=np.float32)
                self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Loaded dataset: {self.dataset_path}")
        print(f"Number of images: {self.num_images}")
        print(f"Image shape: {self.image_shape}")

    def __len__(self):
        return self.num_images

    def _read_image(self, idx):
        if self.data is not None:
            return self.data[idx].copy()

        with h5py.File(self.h5_path, "r") as h5_file:
            image = np.asarray(h5_file[self.dataset_path][idx], dtype=np.float32)

        return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, image):
        if not self.local_norm:
            return image.astype(np.float32)

        mean = np.mean(image)
        std = np.std(image)

        return ((image - mean) / (std + 1e-6)).astype(np.float32)

    def _to_tensor(self, image):
        tensor = torch.from_numpy(image).float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = int(idx)

        if idx < 0:
            idx += self.num_images

        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} is outside the dataset range.")

        second_idx = random.randrange(self.num_images - 1)

        if second_idx >= idx:
            second_idx += 1

        first_image = self._normalize(self._read_image(idx))
        second_image = self._normalize(self._read_image(second_idx))

        first_image = self._to_tensor(first_image)
        second_image = self._to_tensor(second_image)

        return first_image.mT, second_image.mT


'''class ResidualCondeDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', local_norm=True, emb_model=None, dataset_names=None):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split
        self.embs=None
            

        self.data = {}
        self.length = 0
        self.dataset_names=dataset_names
        with h5py.File(h5_path, 'r') as f:
            
            if self.dataset_names is None:
                if isinstance(split, list):
                    self.model_names = split
                elif split == 'train':
                    self.model_names = [n for n in list(f.keys()) if 'Modelo6' not in n]
                elif split == 'test':
                    self.model_names = [n for n in list(f.keys()) if 'Modelo6' in n or 'seap' in n]
                elif split == 'seap':
                    self.model_names = [n for n in list(f.keys()) if 'seap' in n]
                
                else:
                    self.model_names = list(f.keys())
                print(self.model_names)
                self.dataset_names = []
                for model_name in self.model_names:
                    k = list(f[model_name].keys())
                    for kk in k:
                        if 'RGT' not in kk:
                            self.dataset_names.append(f'{model_name}/{kk}')
            

            if local_norm:
                for name in self.dataset_names:
                    
                    data = np.array(f[name][:])
                    z_scores=[]
                    for i, s in enumerate(data):
                        s = np.nan_to_num(s, nan=0.0)
                        mean = np.mean(s)
                        std = np.std(s)

                        z_scores.append((s - mean) / (std + 1e-6))
                            
                    self.data[name] = np.array(z_scores)
                    self.length += len(self.data[name])
            else:
                for name in self.dataset_names:
                    data = np.array(f[name][:])
                    mean = np.mean(data, axis=1, keepdims=True)
                    std = np.std(data, axis=1, keepdims=True)
    
                    z_scores = (data - mean) / (std + 1e-6)
                    self.data[name] = z_scores
    
                    self.length += len(self.data[name])
                    
            if emb_model is not None:
                self.embs = {}
                with torch.no_grad():
                    B = 1024
                    for name in tqdm(self.dataset_names, desc="Extracting Embs..."):
                        data = self.data[name]
                        embs = []
                        for i in range(0, len(data), B):
                            cube = data[i:i+B]
                            cube = torch.tensor(cube).unsqueeze(1).cuda()
                            embs.append(emb_model.encode(cube).cpu())
                
                        self.embs[name] = torch.cat(embs, dim=0)

        self.augment = SeismicAugmentation()
        print(f"Finished loading dataset with {self.length} images")
        


    def __len__(self):
        return self.length

    def get_idx(self, idx):
        for key in self.data.keys():
            n = len(self.data[key])
            if n <= idx:
                idx -= n
            else:
                return self.data[key][idx], key, idx
    def get_random_res(self, exclude=None, exclude_reflectivyty = False):
        candidates = [r for r in self.dataset_names if r != exclude]
        if exclude_reflectivyty:
            candidates = [r for r in candidates if 'Refletividade' not in  r]
        return random.choice(candidates)
        
    def get_random_from_res(self, res, exclude=None):
        n = len(self.data[res])
        seismic = None
        while True:
            idx = random.randint(0, n-1)
            
            if exclude is None or not exclude==idx:
                data = self.data[res][idx]
                return data, idx


    def __getitem__(self, idx):
        
        A, r1, A_idx = self.get_idx(idx)
        is_reflectivity = 'Refletividade' in r1
        P, P_idx = self.get_random_from_res(r1, exclude=A_idx)
        r2 = self.get_random_res(exclude=r1, exclude_reflectivyty = is_reflectivity)
        N, N_idx = self.get_random_from_res(r2)
        
        if self.embs is None:
            return torch.tensor(A).unsqueeze(0), torch.tensor(P).unsqueeze(0), torch.tensor(N).unsqueeze(0)
        else:
            return torch.tensor(A).unsqueeze(0), torch.tensor(P).unsqueeze(0), torch.tensor(N).unsqueeze(0), self.embs[r1][A_idx], self.embs[r1][P_idx], self.embs[r2][N_idx]
'''

class WuFaultDataset(Dataset):
    def __init__(self, h5_path=None, return_fault=True, split="train", augment=None, emb_model=None):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"

        valid_splits = {"train", "test", "all", "wu", "df", "df_train", "df_test"}
        assert split in valid_splits, f"split must be one of {sorted(valid_splits)}, got {split!r}"

        self.file = None
        self.split = split
        self.return_fault = return_fault
        self.augment = augment
        self.emb_model = emb_model

        self.data = []
        self.labels = []
        self.dataset_names = []
        self.label_names = []

        with h5py.File(self.h5_path, "r") as f:
            self.model_names = list(f.keys())

            for model_name in self.model_names:
                is_wu_model = "wu" in model_name.lower()
                is_df_model = "df" in model_name.lower()

                for dataset_name in f[model_name].keys():
                    dataset_name_lower = dataset_name.lower()

                    is_fault = "fault" in dataset_name_lower
                    is_train = "train" in dataset_name_lower
                    is_test = "test" in dataset_name_lower

                    include = (
                        (split == "all")
                        or (split == "train" and is_train)
                        or (split == "test" and is_test)
                        or (split == "wu" and is_wu_model)
                        or (split == "df" and is_df_model)
                        or (split == "df_train" and is_df_model and is_train)
                        or (split == "df_test" and is_df_model and is_test)
                    )

                    if not include:
                        continue

                    full_name = f"{model_name}/{dataset_name}"

                    if is_fault:
                        self.label_names.append(full_name)
                    else:
                        self.dataset_names.append(full_name)

            self.dataset_names.sort()
            self.label_names.sort()

            for name in self.label_names:
                base = f[name][:]
                for s in base:
                    self.labels.append(s)
                    
            for name in self.dataset_names:
                base = f[name][:]
                for s in base:
                    mean = np.mean(s)
                    std = np.std(s)
                    normalized = (s - mean) / (std + 1e-8)
                    self.data.append(normalized[None])
            self.embs = None
            if emb_model is not None:
                embs = []
                with torch.no_grad():
                    B = 1024
                    for i in tqdm(range(0, len(self.data), B)):
                        cube = self.data[i:i+B]
                        cube = torch.tensor(cube).cuda().float()
                        embs.append(emb_model.encode(cube).cpu())
                
                self.embs = torch.cat(embs, dim=0)

            

        print(f"Initialized dataset with {len(self.data)} images")
        self.augment = augment
        
        

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sis = self.data[idx]
        if self.augment is not None:
            sis = self.augment(torch.tensor(sis), per_image=False)
        if self.embs is not None:
            return sis, self.labels[idx], self.embs[idx]
        if self.return_fault:
            return sis, self.labels[idx]
        else:
        
            return sis

class WuResidualCondeDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', emb_model=None):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split
        self.embs=None

            

        self.data = {}
        self.length = 0
        with h5py.File(h5_path, 'r') as f:
            self.model_names = list(f.keys())
            print(self.model_names)
            self.dataset_names = []
            for model_name in self.model_names:
                k = list(f[model_name].keys())
                for kk in k:
                    if 'fault' not in kk:
                        if split in ['train', 'all'] and 'train' in kk:
                            self.dataset_names.append(f'{model_name}/{kk}')
                        if split in ['test', 'all'] and 'test' in kk:
                            self.dataset_names.append(f'{model_name}/{kk}')

            for name in self.dataset_names:
                data = np.array(f[name][:])
                z_scores=[]
                for i, s in enumerate(data):
                    mean = np.mean(s)
                    std = np.std(s)

                    z_scores.append((s - mean) / (std + 1e-6))
                        
                self.data[name] = np.array(z_scores)
                self.length += len(self.data[name])

                    
            if emb_model is not None:
                self.embs = {}
                with torch.no_grad():
                    B = 1024
                    for name in tqdm(self.dataset_names, desc="Extracting Embs..."):
                        data = self.data[name]
                        embs = []
                        for i in range(0, len(data), B):
                            cube = data[i:i+B]
                            cube = torch.tensor(cube).unsqueeze(1).cuda()
                            embs.append(emb_model.encode(cube).cpu())
                
                        self.embs[name] = torch.cat(embs, dim=0)

        self.augment = SeismicAugmentation()
        print(f"Finished loading dataset with {self.length} images")
        


    def __len__(self):
        return self.length

    def get_idx(self, idx):
        for key in self.data.keys():
            n = len(self.data[key])
            if n <= idx:
                idx -= n
            else:
                return self.data[key][idx], key, idx
    def get_random_res(self, exclude=None, exclude_reflectivyty = False):
        candidates = [r for r in self.dataset_names if r != exclude]
        if exclude_reflectivyty:
            candidates = [r for r in candidates if 'Refletividade' not in  r]
        return random.choice(candidates)
        
    def get_random_from_res(self, res, exclude=None):
        n = len(self.data[res])
        seismic = None
        while True:
            idx = random.randint(0, n-1)
            
            if exclude is None or not exclude==idx:
                data = self.data[res][idx]
                return data, idx


    def __getitem__(self, idx):
        
        A, r1, A_idx = self.get_idx(idx)
        is_reflectivity = 'Refletividade' in r1
        P, P_idx = self.get_random_from_res(r1, exclude=A_idx)
        r2 = self.get_random_res(exclude=r1, exclude_reflectivyty = is_reflectivity)
        N, N_idx = self.get_random_from_res(r2)
        
        if self.embs is None:
            return torch.tensor(A).unsqueeze(0), torch.tensor(P).unsqueeze(0), torch.tensor(N).unsqueeze(0)
        else:
            return torch.tensor(A).unsqueeze(0), torch.tensor(P).unsqueeze(0), torch.tensor(N).unsqueeze(0), self.embs[r1][A_idx], self.embs[r1][P_idx], self.embs[r2][N_idx]


class WuOnlyDataset(Dataset):
    def __init__(self, h5_path = None, split = 'train', emb_model=None):
        self.h5_path = h5_path
        assert h5_path is not None, "You must provide a path to the h5 dataset"
        self.file = None
        self.split = split

        # Open once to collect info
        self.data = []
        with h5py.File(h5_path, 'r') as f:
            self.model_names = f.keys()
                
            self.dataset_names = []
            for model_name in self.model_names:
                k = list(f[model_name].keys())
                if 'wu' in model_name:
                    for kk in k:
                        if 'fault' not in kk:
                            if 'train' in kk and split == 'train' or split == 'all':                  
                                self.dataset_names.append(f'{model_name}/{kk}')
                            if 'test' in kk and split == 'test' or split == 'all':
                                self.dataset_names.append(f'{model_name}/{kk}')
                    
            for name in self.dataset_names:
                base = f[name][:]
                for s in base:
                    mean = np.mean(s)
                    std = np.std(s)
                    normalized = (s - mean) / (std + 1e-8)
                    self.data.append(normalized[None])
            self.embs = None
            if emb_model is not None:
                embs = []
                with torch.no_grad():
                    B = 1024
                    for i in tqdm(range(0, len(self.data), B)):
                        cube = self.data[i:i+B]
                        cube = torch.tensor(cube).cuda().float()
                        embs.append(emb_model.encode(cube).cpu())
                
                self.embs = torch.cat(embs, dim=0)

            

        print(f"Initialized dataset with {len(self.data)} images")
        
        

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sis = self.data[idx]
        idx2 = random.randint(0, len(self)-1)
        sis2 = self.data[idx2]
        
        if self.embs is not None:
            return sis, sis2, sis, self.embs[idx], self.embs[idx2], self.embs[idx]

        return sis, sis2, sis
        