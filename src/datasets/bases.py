from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


MASK_STRATEGIES = {"random", "noun", "noun_adj", "agriculture_random"}

# Fruit contains close-up plant/leaf descriptions. The domain vocabulary covers
# plant organs, disease symptoms, color/shape attributes, and visible damage.
FRUIT_AGRICULTURE_TERMS = {
    "agriculture", "plant", "plants", "fruit", "fruits", "leaf", "leaves",
    "stem", "stems", "branch", "branches", "vein", "veins", "surface", "edge",
    "edges", "margin", "tip", "center", "spot", "spots", "lesion", "lesions",
    "speckle", "speckles", "mottle", "mottled", "blotch", "blotches", "patch",
    "patches", "halo", "halos", "rot", "rust", "blight", "mildew", "mold",
    "fungus", "fungal", "disease", "infected", "infection", "damage", "damaged",
    "wilt", "wilted", "wilting", "curl", "curled", "dry", "dried", "necrosis",
    "yellow", "yellowish", "green", "brown", "black", "white", "orange", "red",
    "dark", "pale", "bright", "round", "oval", "elongated", "irregular",
    "serrated", "smooth", "rough", "pointed", "discoloration", "discolored",
    "aphid", "aphids", "mite", "mites", "beetle", "beetles", "pest", "pests",
}

PEST_AGRICULTURE_TERMS = {
    "agriculture", "agricultural", "crop", "crops", "rice", "wheat", "corn",
    "maize", "citrus", "alfalfa", "plant", "plants", "leaf", "leaves", "stem",
    "stems", "branch", "branches", "seed", "seeds", "fruit", "fruits",
    "pest", "pests", "insect", "insects", "bug", "bugs", "aphid", "aphids",
    "beetle", "beetles", "caterpillar", "caterpillars", "cutworm", "cutworms",
    "roller", "rollers", "moth", "moths", "butterfly", "butterflies", "weevil",
    "weevils", "locust", "locusts", "leafhopper", "planthopper", "thrips", "mite",
    "mites", "scale", "borer", "borers", "worm", "worms", "grub", "grubs",
    "larva", "larvae", "fly", "flies", "cicada", "chalcid", "mealybug",
    "whitefly", "head", "thorax", "abdomen", "body", "bodies", "leg", "legs",
    "wing", "wings", "antenna", "antennae", "tail", "segment", "segmented",
    "stripe", "stripes", "spot", "spots", "marking", "markings", "texture",
    "surface", "rough", "smooth", "fuzzy", "scaly", "wrinkled", "cylindrical",
    "elongated", "oval", "curved", "curled", "coiled", "round", "small", "large",
    "green", "brown", "black", "white", "yellow", "orange", "red", "dark", "light",
}

AGRICULTURE_TERMS_BY_DATASET = {
    "FRS-CMR": FRUIT_AGRICULTURE_TERMS,
    "CroDis4CMR": FRUIT_AGRICULTURE_TERMS,
    "FruDis4CMR": FRUIT_AGRICULTURE_TERMS,
    "CCPRD": PEST_AGRICULTURE_TERMS,
    "GjPest4CMR": PEST_AGRICULTURE_TERMS,
}


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("SAG-AlignNet.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 rho=0.3,
                 mask_strategy="random",
                 dataset_name="",
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.rho = rho
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError("Unknown mask strategy: {}".format(mask_strategy))
        self.mask_strategy = mask_strategy
        self.dataset_name = dataset_name
        self.agriculture_terms = AGRICULTURE_TERMS_BY_DATASET.get(
            dataset_name, FRUIT_AGRICULTURE_TERMS)

        self.tokenizer = SimpleTokenizer()
        self.guided_positions = None
        if self.mask_strategy != "random":
            self.guided_positions = self._precompute_guided_positions()

    def _precompute_guided_positions(self):
        """Precompute word-level POS/domain candidates once before workers start."""
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        captions = [sample[3] for sample in self.dataset]
        positions = []
        for caption, doc in zip(captions, nlp.pipe(captions, batch_size=256)):
            encoded = self.tokenizer.encode(caption)[:self.text_length - 2]
            noun_flags, adj_flags, agriculture_flags = [], [], []
            rebuilt = []
            for word in doc:
                word_ids = self.tokenizer.encode(word.text)
                rebuilt.extend(word_ids)
                noun = word.pos_ in {"NOUN", "PROPN"}
                adjective = word.pos_ == "ADJ"
                normalized = word.lemma_.lower().strip()
                surface = word.text.lower().strip()
                agricultural = (normalized in self.agriculture_terms
                                or surface in self.agriculture_terms)
                noun_flags.extend([noun] * len(word_ids))
                adj_flags.extend([adjective] * len(word_ids))
                agriculture_flags.extend([agricultural] * len(word_ids))
            usable = min(len(encoded), len(rebuilt), self.text_length - 2)
            if rebuilt[:usable] != encoded[:usable]:
                noun_flags = noun_flags[:usable] + [False] * (len(encoded) - usable)
                adj_flags = adj_flags[:usable] + [False] * (len(encoded) - usable)
                agriculture_flags = agriculture_flags[:usable] + [False] * (len(encoded) - usable)
            positions.append({
                "noun": {i + 1 for i, flag in enumerate(noun_flags[:len(encoded)]) if flag},
                "noun_adj": {i + 1 for i, flag in enumerate(noun_flags[:len(encoded)]) if flag}
                | {i + 1 for i, flag in enumerate(adj_flags[:len(encoded)]) if flag},
                "agriculture_random": {i + 1 for i, flag in enumerate(
                    agriculture_flags[:len(encoded)]) if flag},
            })
        return positions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        caption_tokens_o = caption_tokens.clone()
        if self.mask_strategy == "random":
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(
                caption_tokens.cpu().numpy(), self.rho)
        else:
            mlm_tokens, mlm_labels = self._build_guided_masked_tokens_and_labels(
                caption_tokens.cpu().numpy(), self.rho,
                self.guided_positions[index][self.mask_strategy])
        mlm_tokens_2, mlm_labels_2 = self._build_random_masked_tokens_and_labels_2(caption_tokens_o.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': mlm_tokens_2,#caption_tokens_o,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens, rho):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        ll = [320, 550, 518, 530, 525, 536, 638, 556, 593, 633, 531, 539, 537, 767, 541, 6190, 2296, 533, 791, 2839, 6998, 11869, 3309, 17982, 9920, 267, 269, 282]
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405 and token not in ll:
                prob = random.random()
                # mask token with 15% probability
                if prob < rho:
                    prob /= rho

                    # 80% randomly change token to mask token
                    if prob < 0.9999:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.99999:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

    def _build_guided_masked_tokens_and_labels(self, tokens, rho, priority_positions):
        """Change only token selection; retain Random's budget and replacement."""
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3))
        excluded = {320, 550, 518, 530, 525, 536, 638, 556, 593, 633, 531, 539,
                    537, 767, 541, 6190, 2296, 533, 791, 2839, 6998, 11869,
                    3309, 17982, 9920, 267, 269, 282}
        candidates = [i for i, token in enumerate(tokens)
                      if 0 < token < 49405 and token not in excluded]
        labels = [0] * len(tokens)
        if not candidates:
            return torch.tensor(tokens), torch.tensor(labels)
        budget = sum(random.random() < rho for _ in candidates)
        budget = max(1, min(budget, len(candidates)))
        priority = [i for i in candidates if i in priority_positions]
        selected = random.sample(priority, min(budget, len(priority)))
        if len(selected) < budget:
            remaining = [i for i in candidates if i not in selected]
            selected.extend(random.sample(remaining, budget - len(selected)))
        for i in selected:
            token = tokens[i]
            prob = random.random()
            if prob < 0.9999:
                tokens[i] = mask
            elif prob < 0.99999:
                tokens[i] = random.choice(token_range)
            labels[i] = token
        return torch.tensor(tokens), torch.tensor(labels)

    def _build_random_masked_tokens_and_labels_2(self, tokens):

        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
