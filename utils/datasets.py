import os
import json
import pickle
from typing import Any, Callable, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.models as models
from torchvision.models.resnet import _resnet, BasicBlock, ResNet


class MiniImageNet(VisionDataset):
    """`MiniImageNet` Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``mini-imagenet-cache-train.pkl`` and `base.pt` will be saved to.
        data_type (str, optional): One of 'base', 'val', 'novel'.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_file = 'base.pth'
    val_file = 'val.pth'
    novel_file = 'novel.pth'
    classes = ['n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081',
               'n01843383', 'n01910747', 'n02074367', 'n02089867', 'n02091831',
               'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915',
               'n02111277', 'n02113712', 'n02120079', 'n02165456', 'n02457408',
               'n02606052', 'n02687172', 'n02747177', 'n02795169', 'n02823428',
               'n02966193', 'n03017168', 'n03047690', 'n03062245', 'n03207743',
               'n03220513', 'n03337140', 'n03347037', 'n03400231', 'n03476684',
               'n03527444', 'n03676483', 'n03838899', 'n03854065', 'n03888605',
               'n03908618', 'n03924679', 'n03998194', 'n04067472', 'n04243546',
               'n04251144', 'n04258138', 'n04275548', 'n04296562', 'n04389033',
               'n04435653', 'n04443257', 'n04509417', 'n04515003', 'n04596742',
               'n04604644', 'n04612504', 'n06794110', 'n07584110', 'n07697537',
               'n07747607', 'n09246464', 'n13054560', 'n13133613', 'n01855672',
               'n02091244', 'n02114548', 'n02138441', 'n02174001', 'n02950826',
               'n02971356', 'n02981792', 'n03075370', 'n03417042', 'n03535780',
               'n03584254', 'n03770439', 'n03773504', 'n03980874', 'n09256479',
               'n01930112', 'n01981276', 'n02099601', 'n02110063', 'n02110341',
               'n02116738', 'n02129165', 'n02219486', 'n02443484', 'n02871525',
               'n03127925', 'n03146219', 'n03272010', 'n03544143', 'n03775546',
               'n04146614', 'n04149813', 'n04418357', 'n04522168', 'n07613480']

    def __init__(
            self,
            root: str,
            data_type: str = 'base',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            load_from_file: bool = True,
    ) -> None:

        super(MiniImageNet, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        assert data_type in ['base', 'val', 'novel']
        self.data_type = data_type  # base, val, or novel set

        self.data: Any = []
        self.targets = []
        if load_from_file:
            self._load_data()
            self._load_meta()

    def set_data(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def split(self, shares) -> Tuple:
        shares_np = np.array([shares]).astype(np.float64)
        shares_np = shares_np / shares_np.sum()

        all_idxs = np.arange(self.data.shape[0])
        split_idxs = tuple([] for _ in shares)
        for i in range(len(self.classes)):
            i_indxs = all_idxs[self.targets == i]
            i_cnts = (len(i_indxs) * shares_np).astype(np.int64)
            i_cnts[0:(i_cnts.sum() - len(i_indxs))] += 1
            assert i_cnts.sum() == len(i_indxs)
            i_cnt_sums = np.cumsum(i_cnts).tolist()
            for j, (st_, end_) in enumerate(zip([0]+i_cnt_sums[:-1], i_cnt_sums)):
                split_idxs[j].append(i_indxs[st_:end_])
        split_idxs = tuple(np.concatenate(idxs_, axis=0) for idxs_ in split_idxs)

        output = []
        for idx_arr in split_idxs:
            partial_ds = MiniImageNet(root=self.root, data_type=self.data_type,
                                      transform=self.transform,
                                      target_transform=self.target_transform,
                                      load_from_file=False)

            partial_ds.set_data(self.data[idx_arr], self.targets[idx_arr])
            output.append(partial_ds)
        return tuple(output)

    def _load_data(self) -> None:

        pt_file = {'base': self.base_file, 'val': self.val_file,
                   'novel': self.novel_file}[self.data_type]
        pt_path = os.path.join(self.root, pt_file)

        if not os.path.exists(pt_path):
            print('Processing...')
            # Let's create these files

            base_pkl_path = os.path.join(self.root, f'mini-imagenet-cache-train.pkl')
            val_pkl_path = os.path.join(self.root, f'mini-imagenet-cache-val.pkl')
            novel_pkl_path = os.path.join(self.root, f'mini-imagenet-cache-test.pkl')

            with open(base_pkl_path, 'rb') as f:
                base_data_pkl = pickle.load(f)
            with open(val_pkl_path, 'rb') as f:
                val_data_pkl = pickle.load(f)
            with open(novel_pkl_path, 'rb') as f:
                novel_data_pkl = pickle.load(f)

            class_names = []
            all_data_pkls = [base_data_pkl, val_data_pkl, novel_data_pkl]
            all_data = tuple(pkl_data['image_data'] for pkl_data in all_data_pkls)
            all_size = tuple(pkl_data['image_data'].shape[0] for pkl_data in all_data_pkls)
            all_lbls = tuple(np.zeros(data_size, dtype=np.int64) for data_size in all_size)
            for pkl_data, lbls in zip(all_data_pkls, all_lbls):
                for cls_name, members_list in pkl_data['class_dict'].items():
                    cls_lbl = len(class_names)
                    class_names.append(cls_name)
                    lbls[members_list] = cls_lbl

            base_lbls, val_lbls, novel_lbls = all_lbls
            base_data, val_data, novel_data = all_data

            if not (class_names == self.classes):
                class_names_str = ''
                for i, cls_name in enumerate(class_names):
                    class_names_str = class_names_str + f"'{cls_name}'," + ('\n' if (i % 5 == 4) else' ')
                raise Exception(f'class_names do not match self.classes. class_names are: [{class_names_str}]')

            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, self.base_file), 'wb') as f:
                torch.save((base_data, base_lbls), f)
            with open(os.path.join(self.root, self.val_file), 'wb') as f:
                torch.save((val_data, val_lbls), f)
            with open(os.path.join(self.root, self.novel_file), 'wb') as f:
                torch.save((novel_data, novel_lbls), f)

        self.data, self.targets = torch.load(os.path.join(self.root, pt_file))

    def _load_meta(self) -> None:
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format(self.data_type)


class TieredImageNet(VisionDataset):
    im_extensions = ('.JPEG', '.jpeg', '.jpg', '.JPG')
    base_classes = ['n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993',
                    'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01675722',
                    'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
                    'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01728572', 'n01728920',
                    'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
                    'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
                    'n01753488', 'n01755581', 'n01756291', 'n01847000', 'n01855032', 'n01855672',
                    'n01860187', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229',
                    'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207',
                    'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110',
                    'n02051845', 'n02056570', 'n02058221', 'n02088094', 'n02088238', 'n02088364',
                    'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379',
                    'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467',
                    'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428',
                    'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258',
                    'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177',
                    'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209',
                    'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413',
                    'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311',
                    'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604',
                    'n02130308', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096',
                    'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577',
                    'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616',
                    'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975',
                    'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166',
                    'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079',
                    'n02497673', 'n02500267', 'n02727426', 'n02793495', 'n02859443', 'n03028079',
                    'n03032252', 'n03457902', 'n03529860', 'n03661043', 'n03781244', 'n03788195',
                    'n03877845', 'n03956157', 'n04081281', 'n04346328', 'n02687172', 'n02690373',
                    'n02692877', 'n02782093', 'n02951358', 'n02981792', 'n03095699', 'n03344393',
                    'n03447447', 'n03662601', 'n03673027', 'n03947888', 'n04147183', 'n04266014',
                    'n04273569', 'n04347754', 'n04483307', 'n04552348', 'n04606251', 'n04612504',
                    'n02979186', 'n02988304', 'n02992529', 'n03085013', 'n03187595', 'n03584254',
                    'n03777754', 'n03782006', 'n03857828', 'n03902125', 'n04392985', 'n02776631',
                    'n02791270', 'n02871525', 'n02927161', 'n03089624', 'n03461385', 'n04005630',
                    'n04200800', 'n04443257', 'n04462240', 'n02799071', 'n02802426', 'n03134739',
                    'n03445777', 'n03598930', 'n03942813', 'n04023962', 'n04118538', 'n04254680',
                    'n04409515', 'n04540053', 'n06785654', 'n02667093', 'n02837789', 'n02865351',
                    'n02883205', 'n02892767', 'n02963159', 'n03188531', 'n03325584', 'n03404251',
                    'n03534580', 'n03594734', 'n03595614', 'n03617480', 'n03630383', 'n03710721',
                    'n03770439', 'n03866082', 'n03980874', 'n04136333', 'n04325704', 'n04350905',
                    'n04370456', 'n04371430', 'n04479046', 'n04591157', 'n02708093', 'n02749479',
                    'n02794156', 'n02841315', 'n02879718', 'n02950826', 'n03196217', 'n03197337',
                    'n03467068', 'n03544143', 'n03692522', 'n03706229', 'n03773504', 'n03841143',
                    'n03891332', 'n04008634', 'n04009552', 'n04044716', 'n04086273', 'n04090263',
                    'n04118776', 'n04141975', 'n04317175', 'n04328186', 'n04355338', 'n04356056',
                    'n04376876', 'n04548280', 'n02672831', 'n02676566', 'n02787622', 'n02804610',
                    'n02992211', 'n03017168', 'n03110669', 'n03249569', 'n03272010', 'n03372029',
                    'n03394916', 'n03447721', 'n03452741', 'n03494278', 'n03495258', 'n03720891',
                    'n03721384', 'n03838899', 'n03840681', 'n03854065', 'n03884397', 'n04141076',
                    'n04311174', 'n04487394', 'n04515003', 'n04536866', 'n02825657', 'n02840245',
                    'n02843684', 'n02895154', 'n03000247', 'n03146219', 'n03220513', 'n03347037',
                    'n03424325', 'n03527444', 'n03637318', 'n03657121', 'n03788365', 'n03929855',
                    'n04141327', 'n04192698', 'n04229816', 'n04417672', 'n04423845', 'n04435653',
                    'n04507155', 'n04523525', 'n04589890', 'n04590129', 'n02910353', 'n03075370',
                    'n03208938', 'n03476684', 'n03627232', 'n03803284', 'n03804744', 'n03874599',
                    'n04127249', 'n04153751', 'n04162706', 'n02951585', 'n03041632', 'n03109150',
                    'n03481172', 'n03498962', 'n03649909', 'n03658185', 'n03954731', 'n03967562',
                    'n03970156', 'n04154565', 'n04208210']
    
    val_classes = ['n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 
                   'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 
                   'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n03207941', 
                   'n03259280', 'n03297495', 'n03483316', 'n03584829', 'n03761084', 'n04070727', 
                   'n04111531', 'n04442312', 'n04517823', 'n04542943', 'n04554684', 'n02791124',
                   'n02804414', 'n02870880', 'n03016953', 'n03018349', 'n03125729', 'n03131574', 
                   'n03179701', 'n03201208', 'n03290653', 'n03337140', 'n03376595', 'n03388549', 
                   'n03742115', 'n03891251', 'n03998194', 'n04099969', 'n04344873', 'n04380533', 
                   'n04429376', 'n04447861', 'n04550184', 'n02666196', 'n02977058', 'n03180011', 
                   'n03485407', 'n03496892', 'n03642806', 'n03832673', 'n04238763', 'n04243546', 
                   'n04428191', 'n04525305', 'n06359193', 'n02966193', 'n02974003', 'n03425413', 
                   'n03532672', 'n03874293', 'n03944341', 'n03992509', 'n04019541', 'n04040759', 
                   'n04067472', 'n04371774', 'n04372370', 'n02701002', 'n02704792', 'n02814533', 
                   'n02930766', 'n03100240', 'n03345487', 'n03417042', 'n03444034', 'n03445924', 
                   'n03594945', 'n03670208', 'n03770679', 'n03777568', 'n03785016', 'n03796401', 
                   'n03930630', 'n03977966', 'n04037443', 'n04252225', 'n04285008', 'n04461696', 
                   'n04467665']

    novel_classes = ['n03314780', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611',
                     'n07613480', 'n07614500', 'n07615774', 'n07697313', 'n07697537', 'n07802026',
                     'n07831146', 'n07836838', 'n07860988', 'n07873807', 'n07875152', 'n07880968',
                     'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n01440764', 'n01443537',
                     'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n02514041',
                     'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379',
                     'n02643566', 'n02655020', 'n02104029', 'n02104365', 'n02105056', 'n02105162',
                     'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030',
                     'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312',
                     'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422',
                     'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063',
                     'n02110185', 'n02110627', 'n02165105', 'n02165456', 'n02167151', 'n02168699',
                     'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856',
                     'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044',
                     'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258',
                     'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02788148',
                     'n02894605', 'n03000134', 'n03160309', 'n03459775', 'n03930313', 'n04239074',
                     'n04326547', 'n04501370', 'n04604644', 'n02795169', 'n02808440', 'n02815834',
                     'n02823428', 'n02909870', 'n02939185', 'n03063599', 'n03063689', 'n03633091',
                     'n03786901', 'n03937543', 'n03950228', 'n03983396', 'n04049303', 'n04398044',
                     'n04493381', 'n04522168', 'n04553703', 'n04557648', 'n04560804', 'n04562935',
                     'n04579145', 'n04591713', 'n09193705', 'n09246464', 'n09256479', 'n09288635',
                     'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597',
                     'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410',
                     'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744',
                     'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275',
                     'n07753592', 'n07754684', 'n07760859', 'n07768694']

    def __init__(
            self,
            root: str = './datasets/tieredimagenet',
            data_type: str = 'val',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            pre_processed_samples: Optional[list] = None
    ) -> None:

        super(TieredImageNet, self).__init__(root, transform=transform, 
                                             target_transform=target_transform)
        self.root = root
        self.data_type = data_type
        self.class_names = {'base': self.base_classes,
                            'val': self.val_classes,
                            'novel': self.novel_classes}[data_type]
        if pre_processed_samples is None:
            self.samples = []
            for cls_idx, cls_name in enumerate(self.class_names):
                cls_dir = f'{self.root}/{cls_name}'
                if not os.path.exists(cls_dir):
                    print(f'*** Warning: Missing tiredimagent directory {cls_dir}')
                    continue
                cls_samples = [(os.path.join(cls_dir, x), cls_idx)
                               for x in os.listdir(cls_dir)
                               if any(x.endswith(im_extension) for im_extension in self.im_extensions)]
                self.samples = self.samples + cls_samples

            self.reorder_samples()
        else:
            self.samples = pre_processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def split(self, shares):
        shares_np = np.array([shares]).astype(np.float64).ravel()
        shares_np = shares_np / shares_np.sum()
        
        targets = np.array([x[1] for x in self.samples])
        all_idxs = np.arange(len(self.samples))
        split_idxs = tuple([] for _ in shares)
        for i in range(len(self.class_names)):
            i_indxs = all_idxs[targets == i]
            i_cnts = (len(i_indxs) * shares_np).astype(np.int64)
            rmng = len(i_indxs) - int(i_cnts.sum())
            i_cnts[0:rmng] += 1
            assert i_cnts.sum() == len(i_indxs), f'Sum({i_cnts}) != {len(i_indxs)}'
            i_cnt_sums = np.cumsum(i_cnts).tolist()
            for j, (st_, end_) in enumerate(zip([0]+i_cnt_sums[:-1], i_cnt_sums)):
                split_idxs[j].append(i_indxs[st_:end_])
        split_idxs = tuple(np.concatenate(idxs_, axis=0) for idxs_ in split_idxs)

        output = []
        for idx_arr in split_idxs:
            partial_ds = TieredImageNet(root=self.root, data_type=self.data_type,
                                        transform=self.transform, target_transform=self.target_transform,
                                        pre_processed_samples=[self.samples[ii] for ii in idx_arr])
            output.append(partial_ds)
        return tuple(output)

    def reorder_samples(self, order_csv=None):
        order_csv = order_csv or f'{self.root}_{self.data_type}_order.json'
        if os.path.exists(order_csv):
            with open(order_csv, 'r') as fp:
                order_list = json.load(fp)
            # e.g. order_list = [['boy', 'male_baby_001.png'], ...]

            cls_file_pairs = list(map(tuple, order_list))
            # e.g. cls_file_pairs = [('boy', 'male_baby_001.png'), ...]
            cfp2idx = {tuple(k): v for v, k in enumerate(cls_file_pairs)}
            cfplen = len(cls_file_pairs)

            def sort_key(sample):
                return cfp2idx.get(tuple(sample[0].split('/')[-2:]), cfplen)
            self.samples.sort(key=sort_key)


class CifarFS(TieredImageNet):
    im_extensions = ('.png',)
    base_classes = ['train', 'skyscraper', 'turtle', 'raccoon', 'spider', 'orange', 'castle',
                    'keyboard', 'clock', 'pear', 'girl', 'seal', 'elephant', 'apple', 'aquarium_fish',
                    'bus', 'mushroom', 'possum', 'squirrel', 'chair', 'tank', 'plate', 'wolf', 'road',
                    'mouse', 'boy', 'shrew', 'couch', 'sunflower', 'tiger', 'caterpillar', 'lion',
                    'streetcar', 'lawn_mower', 'tulip', 'forest', 'dolphin', 'cockroach', 'bear',
                    'porcupine', 'bee', 'hamster', 'lobster', 'bowl', 'can', 'bottle', 'trout',
                    'snake', 'bridge', 'pine_tree', 'skunk', 'lizard', 'cup', 'kangaroo', 'oak_tree',
                    'dinosaur', 'rabbit', 'orchid', 'willow_tree', 'ray', 'palm_tree', 'mountain',
                    'house', 'cloud']
    
    val_classes = ['otter', 'motorcycle', 'television', 'lamp', 'crocodile', 'shark', 'butterfly', 
                   'beaver', 'beetle', 'tractor', 'flatfish', 'maple_tree', 'camel', 'crab', 
                   'sea', 'cattle']
    
    novel_classes = ['baby', 'bed', 'bicycle', 'chimpanzee', 'fox', 'leopard', 'man', 'pickup_truck',
                     'plain', 'poppy', 'rocket', 'rose', 'snail', 'sweet_pepper', 'table', 'telephone',
                     'wardrobe', 'whale', 'woman', 'worm']


class FeaturesDataset(Dataset):
    FEATS_LOADED_CACHE = OrderedDict()
    FEATS_LOADED_CACHE_MAX_BYTES = int(10e9)

    def GET_FEATS_LOADED_CACHE_SIZE(self):
        return sum((x.element_size() * x.nelement() + y.element_size() * y.nelement()) for path, (x, y) in
                   self.FEATS_LOADED_CACHE.items())

    def __init__(self, feat_cache_path, feature_model=None, img_loader=None, device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if isinstance(feat_cache_path, tuple):
            self.data, self.targets = feat_cache_path
        else:
            if not os.path.exists(feat_cache_path):
                print(' => Loading the Raw Images Dataset')
                feat_list = []
                targets_list = []
                print(f' => Producing the Features')
                for i, (img, targ) in enumerate(img_loader):
                    if i % 32 == 0:
                        print(f'[{i:05}] ', end='')
                    img_ = img.to(device)
                    with torch.no_grad():
                        feats = feature_model(img_).squeeze(-1).squeeze(-1).cpu()
                    feat_list.append(feats)
                    targets_list.append(targ)
                    print('.', end='')
                    if i % 32 == 31:
                        print('')
                print('')
                with torch.no_grad():
                    data = torch.cat(feat_list, dim=0).detach().cpu()
                    targets = torch.cat(targets_list, dim=0).detach().cpu()
                print(f' => Storing the Features')
                os.makedirs(os.path.dirname(os.path.abspath(feat_cache_path)), exist_ok=True)
                torch.save((data, targets), feat_cache_path)

            if feat_cache_path in self.FEATS_LOADED_CACHE:
                self.data, self.targets = self.FEATS_LOADED_CACHE[feat_cache_path]
            else:
                data_, targets_ = torch.load(feat_cache_path, map_location='cpu')
                self.FEATS_LOADED_CACHE[feat_cache_path] = data_, targets_
                self.data, self.targets = data_, targets_

            while self.GET_FEATS_LOADED_CACHE_SIZE() > self.FEATS_LOADED_CACHE_MAX_BYTES:
                self.FEATS_LOADED_CACHE.popitem(last=False)
            print(f'Latest cache size : {(self.GET_FEATS_LOADED_CACHE_SIZE() / 1e9):.3f} GB')

        self.transform = None
        self.target_transform = None

    def split(self, shares) -> Tuple:
        shares_np = np.array([shares]).astype(np.float64).ravel()
        shares_np = shares_np / shares_np.sum()

        all_idxs = np.arange(self.data.shape[0])
        split_idxs = tuple([] for _ in shares)
        for i in range(torch.unique(self.targets).numel()):
            i_indxs = all_idxs[self.targets == i]
            i_cnts = (len(i_indxs) * shares_np).astype(np.int64)
            i_cnts[0:(i_cnts.sum() - len(i_indxs))] += 1
            assert i_cnts.sum() == len(i_indxs)
            i_cnt_sums = np.cumsum(i_cnts).tolist()
            for j, (st_, end_) in enumerate(zip([0] + i_cnt_sums[:-1], i_cnt_sums)):
                split_idxs[j].append(i_indxs[st_:end_])
        split_idxs = tuple(np.concatenate(idxs_, axis=0) for idxs_ in split_idxs)

        output = []
        for idx_arr in split_idxs:
            partial_ds = FeaturesDataset((self.data[idx_arr], self.targets[idx_arr]),
                                         feature_model=None, img_loader=None)
            output.append(partial_ds)
        return tuple(output)

    def set_data(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def make_backbone(backbone_arch, ckpt_path, device):
    if backbone_arch.startswith('resnet'):
        return make_resnet_backbone(backbone_arch, ckpt_path, device)
    elif backbone_arch.startswith('densenet') or backbone_arch.startswith('mobilenet'):
        return make_simpleshot_backbone(backbone_arch, ckpt_path, device, norm_type='CL2N')
    else:
        raise ValueError(f'Unknown backbone_arch {backbone_arch}')


def make_resnet_backbone(backbone_arch, ckpt_path, device):
    print(' => Loading the Backbone Model')
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    use_nn_dataparallel = all(key.startswith('module') for key in state_dict)
    model = models.__dict__[backbone_arch]()
    if use_nn_dataparallel:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(state_dict)
    if use_nn_dataparallel:
        model_module = model.module
    else:
        model_module = model
    feature_model = nn.Sequential(*list(model_module.children())[:-1])
    return feature_model


def make_simpleshot_backbone(backbone_arch, ckpt_path, device, norm_type='CL2N'):
    # Examples:
    #   ckpt_path = "/blah/blah/tiered/softmax/densenet121/meanaug_model_best.pth.tar'
    #   backbone_arch= 'densenet121', 'mobilenet84'
    #   dataset_type: 'miniimagenet', 'tieredimagenet'

    # create the model
    from utils import simpleshot_models
    print("=> creating backbone '{}'".format(backbone_arch))
    
    # loading the model state_dict
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
    else:
        raise FileNotFoundError(f"{ckpt_path} does not exist")

    classifier_bias_numel = tuple(checkpoint['state_dict'].values())[-1].numel()
    model = simpleshot_models.__dict__[backbone_arch](num_classes=classifier_bias_numel, remove_linear=False)
    # linear layer is the classifier, it should be true in loading
    print('Number of backbone parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    base_mean = checkpoint['base_mean']
        
    def my_extract_feature(inputs):
        # base_mean is a numpy array
        model.eval()
        with torch.no_grad():
            outputs, _ = model(inputs, True)
            
        outputs = outputs.cpu().data.numpy()
        if norm_type == 'CL2N':
            outputs = outputs - base_mean
            outputs = outputs / np.linalg.norm(outputs, 2, 1)[:, None]
        elif norm_type == 'L2N':
            outputs = outputs / np.linalg.norm(outputs, 2, 1)[:, None]
        
        outputs = torch.tensor(outputs)
        return outputs
    
    return my_extract_feature


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-10 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)


models.__dict__['resnet10'] = resnet10


def get_dali_loader(all_files, all_labels, batch_size=16, num_workers=8, crop_size=224,
                    random_area=None, norm_mean=None, norm_std=None, shuffle=True,
                    do_center_crop=True, resize_size=256):
    if random_area is None:
        random_area = [0.08, 1.00]
    if norm_mean is None:
        norm_mean = [0.485, 0.456, 0.406]
    if norm_std is None:
        norm_std = [0.229, 0.224, 0.225]

    norm_mean = (np.array(norm_mean) * 255).tolist()
    norm_std = (np.array(norm_std) * 255).tolist()
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline

    class HybridTrainPipe(Pipeline):
        def __init__(self, files, labels, batch_size, num_threads, device_id, crop_size, shard_id=0, num_shards=1):
            super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12345 + device_id)
            self.input = ops.FileReader(files=files, labels=labels, shard_id=shard_id, num_shards=num_shards,
                                        random_shuffle=shuffle)
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            self.do_center_crop = do_center_crop
            if self.do_center_crop:
                assert random_area is None
                self.res = ops.Resize(device="gpu", resize_x=resize_size, resize_y=resize_size)
                self.crop = ops.Crop(device="gpu", size=crop_size)
            else:
                assert resize_size is None
                self.rescrop = ops.RandomResizedCrop(device="gpu", size=crop_size, random_area=random_area)
            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                image_type=types.RGB,
                                                mean=norm_mean,
                                                std=norm_std)

        def define_graph(self):
            self.jpegs, self.labels = self.input(name="Reader")
            images = self.decode(self.jpegs)
            if self.do_center_crop:
                images = self.res(images)
                images = self.crop(images)
            else:
                images = self.rescrop(images)
            output = self.cmnp(images)
            return [output, self.labels]

    class DALIDataloader(DALIGenericIterator):
        def __init__(self, pipeline, size, batch_size, auto_reset=True):
            self.mysize = size
            self.batch_size = batch_size
            super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=["data", "label"])

        def __next__(self):
            if self._first_batch is not None:
                batch = self._first_batch
                self._first_batch = None
            else:
                batch = super().__next__()
            data = batch[0]
            return (data["data"], data["label"].squeeze().long())

        def __len__(self):
            return self.mysize // self.batch_size + int(self.mysize % self.batch_size > 0)

    pip_train = HybridTrainPipe(all_files, all_labels, batch_size=batch_size, num_threads=num_workers,
                                device_id=0, crop_size=crop_size, num_shards=1, shard_id=0)
    train_loader = DALIDataloader(pipeline=pip_train, size=len(all_files), batch_size=batch_size)
    return train_loader


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    settings = [(ds, f'resnet{bb}', dt) 
                for ds in ['miniimagenet', 'cifarfs', 'tieredimagenet']
                for bb in [10, 18, 34, 50, 101] 
                for dt in ['base', 'val', 'novel']]
    
    settings += [('tieredimagenet', bb, dt)
                 for bb in ['densenet121', 'mobilenet84']
                 for dt in ['base', 'val', 'novel']]

    for dataset_name, backbone_arch, data_type in settings:
        feat_cache_path = f'../features/{dataset_name}_{data_type}_{backbone_arch}.pth'
        ckpt_path = f'../backbones/{dataset_name}_{backbone_arch}.pth.tar'
        data_root = f'../datasets/{dataset_name}'

        if not os.path.exists(feat_cache_path):
            print(f'Generating {feat_cache_path}.')
            norm_img_mean = [0.485, 0.456, 0.406]
            norm_img_std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=norm_img_mean, std=norm_img_std)

            if backbone_arch.startswith('resnet'):
                resize_pixels = 256
                crop_pixels = 224
            elif backbone_arch in ('densenet121', 'mobilenet84'):
                enlarge = False  # From SimpleShot
                resize_pixels = int(84*256./224.) if enlarge else 84
                crop_pixels = 84
            else:
                raise ValueError(f'backbone {backbone_arch} transformation not implemented.')

            eval_transforms = transforms.Compose([transforms.Resize(resize_pixels),
                                                  transforms.CenterCrop(crop_pixels),
                                                  transforms.ToTensor(),
                                                  normalize])

            if dataset_name == 'miniimagenet':
                imgset = MiniImageNet(root=data_root, data_type=data_type, transform=eval_transforms)
            elif dataset_name == 'tieredimagenet':
                imgset = TieredImageNet(root=data_root, data_type=data_type, transform=eval_transforms)
            elif dataset_name == 'cifarfs':
                imgset = CifarFS(root=data_root, data_type=data_type, transform=eval_transforms)
            else:
                raise ValueError(f'Unknown dataset {dataset_name}')

            img_loader = torch.utils.data.DataLoader(imgset, batch_size=32, shuffle=False, 
                                                     num_workers=12, prefetch_factor=48, 
                                                     pin_memory=False)

            feature_model = make_backbone(backbone_arch=backbone_arch, ckpt_path=ckpt_path, device=device)
        else:
            print(f'{feat_cache_path} already exists. I will not overwrite it.') 
            img_loader = None
            feature_model = None
        
        ###############################
        featset = FeaturesDataset(feat_cache_path, feature_model=feature_model, 
                                  img_loader=img_loader, device=device)
