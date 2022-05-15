import torch
import torch.nn.functional as F
import numpy as np

def mean_list(l):
    if len(l) == 0:
        return 0
    else:
        return np.mean(np.array(l))


class JaccardIOUMetric(object):
    def __init__(self, dtype='numpy'):
        self.insec_list = []
        self.union_list = []
        self.global_iou = 0
        self.instance_iou_list = []
        self.instance_iou = 0
        self.dtype = dtype
        self.iou_bins_list = [[] for _ in range(0, 10)]
        self.iou_bins_mean = []
        assert dtype in ('torch', 'numpy')
    
    def reset(self, dtype='numpy'):
        self.__init__(dtype)

    def jaccard_iou(self, prediction, target):
        if self.dtype == 'numpy':
            insec = np.sum(prediction & target).item()
            union = np.sum(prediction | target).item()
        else:
            insec = torch.sum(prediction & target).item()
            union = torch.sum(prediction | target).item()
        if union == 0: insec = 0; union=1
        return (insec, union)

    def update(self, prediction, target, visible_ratio=None):
        assert prediction.ndim == target.ndim
        assert 2 <= prediction.ndim <= 3

        if prediction.ndim == 2:
            if self.dtype == 'torch':
                prediction = prediction.unsqueeze(0)
                target = target.unsqueeze(0)
            else:
                prediction = prediction[None, ...]
                target = target[None, ...]

        local_insec, local_union = [], []
        for i in range(len(prediction)):
            insec, union = self.jaccard_iou(prediction[i], target[i])
            local_insec.append(insec)
            local_union.append(union)
        self.insec_list.extend(local_insec)
        self.union_list.extend(local_union)
        
        self.global_iou = np.sum(self.insec_list)/np.sum(self.union_list)
        self.instance_iou_list = np.array(self.insec_list)/np.array(self.union_list)
        self.instance_iou = np.mean(self.instance_iou_list)
        
        if visible_ratio is not None:
            visible_ratio = visible_ratio.cpu().numpy()
            visible_ratio_index = np.clip((visible_ratio*10).astype(np.int), a_min=0, a_max=9)
            local_instance_iou = np.array(local_insec)/np.array(local_union)
            for vridx, iou in zip(visible_ratio_index, local_instance_iou):
                self.iou_bins_list[vridx].append(iou.item())
            self.iou_bins_mean = [mean_list(x) for x in self.iou_bins_list]
            self.iou_bins_str = np.array2string(np.array(self.iou_bins_mean), precision=2)

    def export(self, ks):
        return [self.__dict__[k] for k in ks]


class FIDScoreMetric(object):
    def __init__(self, extractor):
        self.extractor = extractor
        self.extractor.eval()
        self.target_feat = []
        self.pred_feat = []
    
    def get_feat(self, input_th):
        feat = self.extractor(input_th)[0]
        if feat.size(2) != 1 or feat.size(3) != 1:
            feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1))
        feat = feat.cpu().numpy().reshape(feat.size(0), -1)
        return feat

    def get_activation_statistics(self, feat):
        mu = np.mean(feat, axis=0)
        sigma = np.cov(feat, rowvar=False)
        return (mu, sigma)

    def update(self, prediction, target):
        assert prediction.ndim == target.ndim
        assert prediction.ndim == 4    # NCHW

        with torch.no_grad():
            pred_feat = self.get_feat(prediction)
            target_feat = self.get_feat(target)
            self.pred_feat.append(pred_feat)
            self.target_feat.append(target_feat)

    def get_frechet_distance(self, mu_sigma1, mu_sigma2):
        mu1, sigma1 = mu_sigma1
        mu2, sigma2 = mu_sigma2
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        from scipy import linalg
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def export(self, ks):
        self.pred_feat = np.concatenate(self.pred_feat, axis=0)
        self.target_feat = np.concatenate(self.target_feat, axis=0)
        self.mu_sigma1 = self.get_activation_statistics(self.pred_feat)
        self.mu_sigma2 = self.get_activation_statistics(self.target_feat)
        self.global_fid = self.get_frechet_distance(self.mu_sigma1, self.mu_sigma2)
        return [self.__dict__[k] for k in ks]



if __name__ == "__main__":
    # import numpy as np
    # ioumetricer = JaccardIOUMetric('numpy')
    # for i in range(100):
    #     aa = np.random.random((1, 20, 20)) > 0.5
    #     bb = np.random.random((1, 20, 20)) > 0.5
    #     ioumetricer.update(aa, bb)
    
    # print (ioumetricer.export(['global_iou', 'instance_iou']))
    from PIL import Image
    from backbone.inception import InceptionV3

    inception = InceptionV3(pretrain_path='pretrains/pt_inception-2015-12-05-6726825d.pth')

    def imread(filename):
        return np.asarray(Image.open(filename).convert('RGB').resize((100, 100)), dtype=np.uint8)[..., :3]

    inception.cuda()
    fsm = FIDScoreMetric(inception)
    pred = imread("/data00/zhouqiang.madamada/research/pytorch-fid/images1/2007_000027.jpg")[None, ...]
    gt = imread("/data00/zhouqiang.madamada/research/pytorch-fid/images2/2008_001004.jpg")[None, ...]
    pred = torch.from_numpy(pred.transpose((0, 3, 1, 2)) / 255).type(torch.FloatTensor).cuda()
    gt = torch.from_numpy(gt.transpose((0, 3, 1, 2)) / 255).type(torch.FloatTensor).cuda()
    fsm.update(pred, gt)
    print (fsm.export(["fid", "mu_sigma1", "mu_sigma2"]))



