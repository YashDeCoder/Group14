import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

class visualize:
    def plot_roc_curve(frame_prob, true_labels, class_labels):
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(true_labels == class_label, frame_prob[class_label])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_label} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(frame_prob, true_labels, class_labels):
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            precision, recall, _ = precision_recall_curve(true_labels == class_label, frame_prob[class_label])
            avg_precision = average_precision_score(true_labels == class_label, frame_prob[class_label])
            plt.plot(recall, precision, label=f'Class {class_label} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()

    def plot_calibration_curve(frame_prob, true_labels, class_labels):
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            prob_true, prob_pred = calibration_curve(true_labels == class_label, frame_prob[class_label], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=f'Class {class_label}')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend(loc='lower right')
        plt.show()

    def plot_probability_histogram(frame_prob, class_labels, set_type='Training'):
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(class_labels):
            plt.hist(frame_prob[class_label], bins=10, alpha=0.5, label=f'Class {class_label}')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Predicted Probabilities ({set_type} Set)')
        plt.legend(loc='upper right')
        plt.show()
