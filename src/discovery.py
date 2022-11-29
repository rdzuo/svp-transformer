import numpy
import math
import torch
from sklearn.cluster import KMeans

def shapelet_discovery(self, X, train_labels, cluster_num, batch_size = 50):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        '''

        slide_num = 3
        alpha = 0.6
        beta = 6
        count = 0
        X_slide_num = []

        for m in range(slide_num):
            # slide the raw time series and the corresponding class and variate label
            X_slide, candidates_dim, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
            X_slide_num.append(numpy.shape(X_slide)[0])
            beta =  beta -2
            alpha = beta/10

            test = utils.Dataset(X_slide)
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size)

            self.encoder = self.encoder.eval()

            # encode slide TS
            with torch.no_grad():
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # 2D to 3D
                    batch.unsqueeze_(1)
                    batch = self.encoder(batch)

                    if count == 0:
                        representation = batch
                    else:
                        representation = numpy.concatenate((representation, batch), axis=0)
                    count += 1
            self.encoder = self.encoder.train()
            count = 0
            # concatenate the new representation from different slides
            if m == 0 :
                representation_all = representation
                representation_dim = candidates_dim
                representation_class_label = candidates_class_label
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis = 0)
                representation_dim = representation_dim + candidates_dim
                representation_class_label = numpy.concatenate((representation_class_label, candidates_class_label), axis=0)

        # cluster all the new representations
        num_cluster = cluster_num
        kmeans = KMeans(n_clusters = num_cluster)
        kmeans.fit(representation_all)

        # init candidate as list
        candidate = []
        candidate_dim = numpy.zeros(num_cluster)

        # select the nearest to the centroid
        for i in range(num_cluster):
            dim_in_cluster_i = list()
            class_label_cluster_i = list()
            dist = math.inf
            for j in range(representation_all[kmeans.labels_==i][:,0].size):
                match_full = numpy.where(representation_all == (representation_all[kmeans.labels_==i][j]))
                match = numpy.unique(match_full)
                dist_tmp = numpy.linalg.norm(representation_all[kmeans.labels_==i][j] - kmeans.cluster_centers_[i])
                for k in range(match.shape[0]):
                    dim_in_cluster_i.append(representation_dim[match[k]])
                    class_label_cluster_i.append(representation_class_label[match[k]])
                if dist_tmp < dist:
                    dist = dist_tmp

                    # trace back the original candidates
                    nearest = numpy.where(representation_all == (representation_all[kmeans.labels_==i][j]))
                    sum_X_slide_num = 0
                    for k in range(slide_num):
                        sum_X_slide_num += X_slide_num[k]
                        if (nearest[0][0] < sum_X_slide_num):
                            index_slide = nearest[0][0] - sum_X_slide_num + X_slide_num[k]
                            X_slide_disc = slide.slide_MTS_dim(X, (0.6-k*0.2))
                            candidate_tmp = X_slide_disc[index_slide]
                            candidate_dim[i] = index_slide % numpy.shape(X)[1]
                            break
            class_label_top = (Counter(class_label_cluster_i).most_common(1)[0][1] / len(class_label_cluster_i))
            dim_label_top = (Counter(dim_in_cluster_i).most_common(1)[0][1] / len(dim_in_cluster_i))
            if (class_label_top < (1/numpy.unique(train_labels).shape[0])) or (dim_label_top < (1/numpy.shape(X)[1])) :
                del candidate_dim[-1]
                continue
            # list append method
            candidate.append(candidate_tmp)

        return candidate, candidate_dim