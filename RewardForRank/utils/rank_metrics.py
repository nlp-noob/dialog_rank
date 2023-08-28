from sklearn.metrics import average_precision_score


class RankEvaluator():
    def get_p_labels_cnt(self, labels:list):
        p_cnt = 0
        for label in labels:
            if label == 1:
                p_cnt += 1
        return p_cnt
    def get_map(self, true_labels: list, scores: list):
        # 1 for true 0 for false
        label_score_pairs = [[score, true_label] for true_label, score in zip(true_labels, scores)]
        sorted_pairs = sorted(label_score_pairs, key=lambda x: x[0], reverse=True)
        ap_sum = 0
        true_cnt = 0
        for i, pair in enumerate(sorted_pairs):
            check_cnt = i + 1
            if pair[1] == 1:
                true_cnt += 1
                ap_sum += true_cnt / check_cnt
        return ap_sum / true_cnt

    def _collate_groups(self, true_labels, scores, group_id_list):
        true_labels_groups = []
        scores_groups = []
        last_group_id = None
        true_labels_group = []
        scores_group = []

        for true_label, score, group_id in zip(true_labels, scores, group_id_list):

            if last_group_id is None or last_group_id == group_id:
                true_labels_group.append(true_label)
                scores_group.append(score)
            else:
                true_labels_groups.append(true_labels_group)
                scores_groups.append(scores_group)
                true_labels_group = [true_label]
                scores_group = [score]

            last_group_id = group_id
        if len(scores_group) > 0:
            true_labels_groups.append(true_labels_group)
            scores_groups.append(scores_group)
        return true_labels_groups, scores_groups

    def get_top_k_acc_list(self, true_labels:list, scores:list, top_k_list:list, group_id_list:list = None):
        if group_id_list is None:
            group_id_list = [0 for i in range(len(scores))]

        true_labels_groups, scores_groups = self._collate_groups(true_labels, scores, group_id_list)

        top_k_acc_list_total = [0 for i in range(len(top_k_list))]

        for true_labels_group, scores_group in zip(true_labels_groups, scores_groups):

            label_score_pairs = [[score, true_label] for true_label, score in zip(true_labels_group, scores_group)]
            sorted_pairs = sorted(label_score_pairs, key=lambda x: x[0], reverse=True)
            sorted_labels =  [pair[1] for pair in sorted_pairs]
            p_cnt = self.get_p_labels_cnt(sorted_labels)
            top_k_acc_list = []
            for top_k in top_k_list:
                top_k_p_cnt = 0
                out_of_range = False
                for i in range(top_k):
                    if i >= len(sorted_labels):
                        out_of_range = True
                        break
                    if sorted_labels[i] == 1:
                        top_k_p_cnt += 1
                if out_of_range:
                    top_k_acc_list.append(1)
                elif top_k >= p_cnt:
                    top_k_acc = top_k_p_cnt / p_cnt
                    top_k_acc_list.append(top_k_acc)
                else:
                    top_k_acc = top_k_p_cnt / top_k
                    top_k_acc_list.append(top_k_acc)
            for top_k_idx, top_k_acc_value in enumerate(top_k_acc_list):
                top_k_acc_list_total[top_k_idx] += top_k_acc_value

        mean_top_k_acc_list = [acc / len(scores_groups) for acc in top_k_acc_list_total]

        return mean_top_k_acc_list


def test():
    rank_evaluator = RankEvaluator()
    true_labels = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    scores = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    group_id = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    print(rank_evaluator.get_map(true_labels, scores))


if __name__ == "__main__":
    test()
