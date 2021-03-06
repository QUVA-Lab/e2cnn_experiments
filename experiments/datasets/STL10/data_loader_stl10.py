
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset, Subset


DATA_DIR = "./datasets/STL10/"

CIFAR_MEAN = np.array([125.3, 123.0, 113.9]) / 255.0  # = np.array([0.49137255, 0.48235294, 0.44666667])
CIFAR_STD = np.array([63.0, 62.1, 66.7]) / 255.0  # = np.array([0.24705882, 0.24352941, 0.26156863])
MEAN = np.array([0.44508205, 0.43821473, 0.40541945])
STD = np.array([0.26199411, 0.25827974, 0.27239384])


########################################################################################################################
# The following piece of code was adapted from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# which has the following ECL-2.0 license:
########################################################################################################################
# Educational Community License, Version 2.0 (ECL-2.0)
#
# Version 2.0, April 2007
#
# http://www.osedu.org/licenses/
#
# The Educational Community License version 2.0 ("ECL") consists of the Apache 2.0 license, modified to change the scope of the patent grant in section 3 to be specific to the needs of the education communities using this license. The original Apache 2.0 license can be found at: http://www.apache.org/licenses /LICENSE-2.0
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
# "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.
#
# "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.
#
# "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
#
# "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.
#
# "Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.
#
# "Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.
#
# "Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).
#
# "Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.
#
# "Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
#
# "Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.
#
# 2. Grant of Copyright License.
#
# Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.
#
# 3. Grant of Patent License.
#
# Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed. Any patent license granted hereby with respect to contributions by an individual employed by an institution or organization is limited to patent claims where the individual that is the author of the Work is also the inventor of the patent claims licensed, and where the organization or institution has the right to grant such license under applicable grant and research funding agreements. No other express or implied licenses are granted.
#
# 4. Redistribution.
#
# You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:
#
#     You must give any other recipients of the Work or Derivative Works a copy of this License; and You must cause any modified files to carry prominent notices stating that You changed the files; and You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License. You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.
#
# 5. Submission of Contributions.
#
# Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.
#
# 6. Trademarks.
#
# This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.
#
# 7. Disclaimer of Warranty.
#
# Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
#
# 8. Limitation of Liability.
#
# In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
#
# 9. Accepting Warranty or Additional Liability.
#
# While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
#
# END OF TERMS AND CONDITIONS
#
# APPENDIX: How to apply the Educational Community License to your work
#
# To apply the Educational Community License to your work, attach
# the following boilerplate notice, with the fields enclosed by
# brackets "[]" replaced with your own identifying information.
# (Don't include the brackets!) The text should be enclosed in the
# appropriate comment syntax for the file format. We also recommend
# that a file or class name and description of purpose be included on
# the same "printed page" as the copyright notice for easier
#     identification within third-party archives.
#
# Copyright [yyyy] [name of copyright owner] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.osedu.org/licenses /ECL-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the license.
class Cutout:
    """Randomly mask out a patch from an image.
    Args:
        size (int): The size of the square patch.
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image
        Returns:
            Tensor: Image with a hole of dimension size x size cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)
        
        mask[y1: y2, x1: x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

########################################################################################################################


def __build_stl10_loaders(batch_size,
                          eval_batchsize,
                          validation=True,
                          num_workers=8,
                          augment=False,
                          reshuffle=True,
                          mean=MEAN,
                          std=STD,
                          ):
    
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    
    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Cutout(32),
            Cutout(60),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # Cutout(24),
            Cutout(48),
            normalize,
        ])
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    
    # load the dataset
    train_dataset = datasets.STL10(
        root=DATA_DIR, split="train",
        download=True, transform=train_transform,
    )
    
    test_dataset = datasets.STL10(
        root=DATA_DIR, split="test",
        download=True, transform=valid_transform,
    )

    if validation:
        
        valid_dataset = datasets.STL10(
            root=DATA_DIR, split="train",
            download=True, transform=valid_transform,
        )
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        
        if reshuffle:
            np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=eval_batchsize, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batchsize, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    n_inputs = 3
    n_classes = 10
    
    return train_loader, valid_loader, test_loader, n_inputs, n_classes


def build_stl10_loaders(batch_size,
                        eval_batchsize,
                        validation=True,
                        num_workers=8,
                        augment=False,
                        reshuffle=True,
                        ):
    return __build_stl10_loaders(batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                 mean=MEAN, std=STD)


def build_stl10cif_loaders(batch_size,
                           eval_batchsize,
                           validation=True,
                           num_workers=8,
                           augment=False,
                           reshuffle=True,
                           ):
    return __build_stl10_loaders(batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                 mean=CIFAR_MEAN, std=CIFAR_STD)


def __balanced_subdataset_idxs(train_size, validation_size, labels, reshuffle):
        num_train = len(labels)
        assert train_size + validation_size <= num_train

        classes = set(labels)
        
        labels_idxs = {c: list() for c in classes}
        ratios = {c: 0. for c in classes}
        
        for i, l in enumerate(labels):
            labels_idxs[l].append(i)
            ratios[l] += 1.
        
        train_idx = list()
        valid_idx = list()
        for c in classes:
            ratios[c] /= num_train
            
            if reshuffle:
                np.random.shuffle(labels_idxs[c])
            
            ts = int(round(train_size * ratios[c]))
            vs = int(round(validation_size * ratios[c]))

            valid_idx += labels_idxs[c][:vs]
            train_idx += labels_idxs[c][vs:vs+ts]
        
        return train_idx, valid_idx
        
        
def __build_stl10_frac_loaders(size,
                               batch_size,
                               eval_batchsize,
                               validation=True,
                               num_workers=8,
                               augment=False,
                               reshuffle=True,
                               mean=MEAN,
                               std=STD,
                               ):
    
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    
    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Cutout(32),
            Cutout(60),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # Cutout(24),
            Cutout(48),
            normalize,
        ])
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    
    # load the dataset
    train_dataset = datasets.STL10(
        root=DATA_DIR, split="train",
        download=True, transform=train_transform,
    )
    
    test_dataset = datasets.STL10(
        root=DATA_DIR, split="test",
        download=True, transform=valid_transform,
    )
    
    if validation:
        
        valid_dataset = datasets.STL10(
            root=DATA_DIR, split="train",
            download=True, transform=valid_transform,
        )
        
        validation_size = 1000
        train_idx, valid_idx = __balanced_subdataset_idxs(size, validation_size, train_dataset.labels, reshuffle)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=eval_batchsize, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
    
        train_idx, _ = __balanced_subdataset_idxs(size, 0, train_dataset.labels, reshuffle)
    
        train_sampler = SubsetRandomSampler(train_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = None
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batchsize, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    n_inputs = 3
    n_classes = 10
    
    return train_loader, valid_loader, test_loader, n_inputs, n_classes


def build_stl10_frac_loaders(size,
                             batch_size,
                             eval_batchsize,
                             validation=True,
                             num_workers=8,
                             augment=False,
                             reshuffle=True,
                             ):
    return __build_stl10_frac_loaders(size, batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                      mean=MEAN, std=STD)


def build_stl10cif_frac_loaders(size,
                                batch_size,
                                eval_batchsize,
                                validation=True,
                                num_workers=8,
                                augment=False,
                                reshuffle=True,
                                ):
    return __build_stl10_frac_loaders(size, batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                 mean=CIFAR_MEAN, std=CIFAR_STD)

