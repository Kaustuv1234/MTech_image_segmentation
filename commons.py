import os 
import numpy as np 
import torch 
import torch.nn as nn 

# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from modules import fpn 
from utils import *

import warnings
warnings.filterwarnings('ignore')

def get_model_and_optimizer(args, logger):
    # Init model 
    model = fpn.PanopticFPN(args) # pantopic feature pyramid network
    model = nn.DataParallel(model) # splits data automatically and sends job orders to multiple models on several GPUs
    model = model.cuda() # send model to GPU

    # Init classifier (for eval only.)
    # Conv2d layer that outputs K channels(K as in K-means). It has been parallelized and present on GPU.
    classifier = initialize_classifier(args)

    # Init optimizer 
    if args.optim_type == 'SGD':
        logger.info('SGD optimizer is used.')
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'Adam':
        logger.info('Adam optimizer is used.')
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.lr)

    # optional restart. 
    args.start_epoch  = 0 
    if args.restart or args.eval_only: 
        load_path = os.path.join(args.save_model_path, 'checkpoint.pth.tar')
        if args.eval_only:
            load_path = args.eval_path
        if os.path.isfile(load_path):
            checkpoint  = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            classifier.load_state_dict(checkpoint['classifier1_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint. [epoch {}]'.format(args.start_epoch))
        else:
            logger.info('No checkpoint found at [{}].\nStart from beginning...\n'.format(load_path))
    
    return model, optimizer, classifier



def run_mini_batch_kmeans(args, logger, dataloader, model, view):
    """
    num_init_batches: (int) The number of batches/iterations to accumulate before the initial k-means clustering.
    num_batches     : (int) The number of batches/iterations to accumulate before the next update. 
    """
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.K_train)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    
    # Choose which view it is now. 
    dataloader.dataset.view = view

    model.train()
    with torch.no_grad():
        for i_batch, (indice, image) in enumerate(dataloader):
            # 1. Compute initial centroids from the first few batches. 
            if view == 1:
                image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                features = model(image)
            elif view == 2:
                image = image.cuda(non_blocking=True)
                features = eqv_transform_if_needed(args, dataloader, indice, model(image))
            else:
                # For evaluation. 
                image = image.cuda(non_blocking=True)
                features = model(image)

            # Normalize.
            if args.metric_test == 'cosine':
                features = F.normalize(features, dim=1, p=2)
            
            if i_batch == 0:
                logger.info('Batch input size : {}'.format(list(image.shape)))
                logger.info('Batch feature : {}'.format(list(features.shape)))
            
            features = feature_flatten(features).detach().cpu()
            if num_batches < args.num_init_batches:
                featslist.append(features)
                num_batches += 1
                
                if num_batches == args.num_init_batches or num_batches == len(dataloader):
                    if first_batch:
                        # Compute initial centroids. 
                        # By doing so, we avoid empty cluster problem from mini-batch K-Means. 
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.K_train, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)

                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                    else:
                        b_feat = torch.cat(featslist)
                        faiss_module = module_update_centroids(faiss_module, centroids)
                        D, I = faiss_module.search(b_feat.numpy().astype('float32'), 1)

                        kmeans_loss.update(D.mean())

                        # Update centroids. 
                        for k in np.unique(I):
                            idx_k = np.where(I == k)[0]
                            data_count[k] += len(idx_k)
                            centroid_lr    = len(idx_k) / (data_count[k] + 1e-6)
                            centroids[k]   = (1 - centroid_lr) * centroids[k] + centroid_lr * b_feat[idx_k].mean(0).numpy().astype('float32')
                    
                    # Empty. 
                    featslist   = []
                    num_batches = args.num_init_batches - args.num_batches

            if (i_batch % 100) == 0:
                logger.info('[Saving features]: {} / {} | [K-Means Loss]: {:.4f}'.format(i_batch, len(dataloader), kmeans_loss.avg))

    centroids = torch.tensor(centroids, requires_grad=False).cuda()

    return centroids, kmeans_loss.avg




def compute_labels(args, logger, dataloader, model, centroids, view):
    """
    Label all images for each view with the obtained cluster centroids. 
    The distance is efficiently computed by setting centroids as convolution layer. 
    """
    K = centroids.size(0)

    # Choose which view it is now. 
    dataloader.dataset.view = view

    # Define metric function with conv layer. 
    metric_function = get_metric_as_conv(centroids)

    counts = torch.zeros(K, requires_grad=False).cpu()
    model.eval()
    with torch.no_grad():
        for i, (indice, image) in enumerate(dataloader):
            if view == 1:
                image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
                features = model(image)
            elif view == 2:
                image = image.cuda(non_blocking=True)
                features = eqv_transform_if_needed(args, dataloader, indice, model(image))

            # Normalize.
            if args.metric_train == 'cosine':
                features = F.normalize(features, dim=1, p=2)

            B, C, H, W = features.shape
            if i == 0:
                logger.info('Centroid size      : {}'.format(list(centroids.shape)))
                logger.info('Batch input size   : {}'.format(list(image.shape)))
                logger.info('Batch feature size : {}\n'.format(list(features.shape)))

            # Compute distance and assign label. 
            scores  = compute_negative_euclidean(features, centroids, metric_function) 

            # Save labels and count. 
            for idx, idx_img in enumerate(indice):
                counts += postprocess_label(args, K, idx, idx_img, scores, n_dual=view)

            if (i % 200) == 0:
                logger.info('[Assigning labels] {} / {}'.format(i, len(dataloader)))
    weight = counts / counts.sum()

    return weight


def evaluate(args, logger, dataloader, classifier, model):
    logger.info('====== METRIC TEST : {} ======\n'.format(args.metric_test))
    histogram = np.zeros((args.K_test, args.K_test)) # 27 x 27 by default

    # set to evaluation mode
    model.eval()
    classifier.eval()

    with torch.no_grad(): # disables gradient calculation, useful for inference, reduces memory consumption for computations 
        
        for i, (_, image, label) in enumerate(dataloader):
            image = image.cuda(non_blocking=True) # move image to GPU in background
            features = model(image) # get feature vector [batch, channel, height, width] 


            if args.metric_test == 'cosine':
                features = F.normalize(features, dim=1, p=2) # convert into unit vector
            
            B, C, H, W = features.size()
            if i == 0:
                logger.info('Batch input size   : {}'.format(list(image.shape)))
                logger.info('Batch label size   : {}'.format(list(label.shape)))
                logger.info('Batch feature size : {}\n'.format(list(features.shape)))

            # get probabilities
            probs = classifier(features) 

            # upsample the probabilities to match label shape
            probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False) 

            # get indices of top k values
            preds = probs.topk(1, dim=1)[1].view(B, -1).cpu().numpy()

            # 
            label = label.view(B, -1).cpu().numpy()
            print('probs', probs.shape)
            print('preds', preds.shape)
            print('label', label.shape)

            histogram += scores(label, preds, args.K_test)
            
            if i%20==0:
                logger.info('{}/{}'.format(i, len(dataloader)))
    
    # Hungarian Matching. 
    m = linear_assignment(histogram.max() - histogram)

    # Evaluate. 
    acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum() * 100

    new_hist = np.zeros((args.K_test, args.K_test))
    for idx in range(args.K_test):
        new_hist[m[idx, 1]] = histogram[idx]
    
    # NOTE: Now [new_hist] is re-ordered to 12 thing + 15 stuff classses. 
    res1 = get_result_metrics(new_hist)
    logger.info('ACC  - All: {:.4f}'.format(res1['overall_precision (pixel accuracy)']))
    logger.info('mIOU - All: {:.4f}'.format(res1['mean_iou']))

    # For Table 2 - partitioned evaluation.
    if args.thing and args.stuff:
        res2 = get_result_metrics(new_hist[:12, :12])
        logger.info('ACC  - Thing: {:.4f}'.format(res2['overall_precision (pixel accuracy)']))
        logger.info('mIOU - Thing: {:.4f}'.format(res2['mean_iou']))

        res3 = get_result_metrics(new_hist[12:, 12:])
        logger.info('ACC  - Stuff: {:.4f}'.format(res3['overall_precision (pixel accuracy)']))
        logger.info('mIOU - Stuff: {:.4f}'.format(res3['mean_iou']))
    
    return acc, res1
