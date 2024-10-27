# import os
import torch
# import pickle
import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process reward bench dataset with optional hard subset filtering.")
    parser.add_argument("--filter_by_subset", default="chat_hard", help="Filter for hard subsets in the reward bench dataset.")
    #parser.add_argument("--chat", action="store_true", help="Filter for regular/easy subsets in the reward bench dataset.")
    parser.add_argument("--shorten_size", default="768", help="Take full feature length or only the first 768 dimensions")  
    return parser.parse_args()

# def load_data(base_dir):
#     """Load all required data files from the specified directory"""
#     # Construct full paths

#     args = parse_args()
#     features_chosen_path = os.path.join(base_dir, 'features_chosen_full_length_{}.pkl'.format(args.filter_by_subset))
#     features_rejected_path = os.path.join(base_dir, 'features_rejected_full_length_{}.pkl'.format(args.filter_by_subset))
#     scores_chosen_path = os.path.join(base_dir, 'scores_chosen.pkl')
#     scores_rejected_path = os.path.join(base_dir, 'scores_rejected.pkl')
    
#     # Load each file
#     print("Loading data files...")
#     with open(features_chosen_path, 'rb') as f:
#         features_chosen_4096 = pickle.load(f).to(torch.float32)
#     with open(features_rejected_path, 'rb') as f:
#         features_rejected_4096 = pickle.load(f).to(torch.float32)
#     with open(scores_chosen_path, 'rb') as f:
#         scores_chosen = pickle.load(f).astype(np.float32)
#     with open(scores_rejected_path, 'rb') as f:
#         scores_rejected = pickle.load(f).astype(np.float32)
    
#     # Extract 768d features
#     features_chosen_768 = features_chosen_4096[:, :int(args.shorten_size)]
#     features_rejected_768 = features_rejected_4096[:, :int(args.shorten_size)]
    
#     print(f"\nFeature shapes:")
#     print(f"4096d features - Chosen: {features_chosen_4096.shape}, Rejected: {features_rejected_4096.shape}")
#     print(f"768d features - Chosen: {features_chosen_768.shape}, Rejected: {features_rejected_768.shape}")
#     print(f"Scores - Chosen: {len(scores_chosen)}, Rejected: {len(scores_rejected)}")
    
#     return (features_chosen_768, features_rejected_768, 
#             features_chosen_4096, features_rejected_4096, 
#             scores_chosen, scores_rejected)

# def evaluate_features(features_chosen, features_rejected, scores_chosen, scores_rejected, feature_name=""):
#     """Evaluate feature set performance with train-test split"""
    
#     # Combine features and scores
#     X = np.vstack([features_chosen, features_rejected])
#     y = np.concatenate([scores_chosen, scores_rejected])
    
#     # Create train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # Train regression model
#     reg = LinearRegression()
#     reg.fit(X_train, y_train)
    
#     # Get predictions
#     train_predictions = reg.predict(X_train)
#     test_predictions = reg.predict(X_test)
    
#     # Calculate R² scores
#     train_r2 = r2_score(y_train, train_predictions)
#     test_r2 = r2_score(y_test, test_predictions)
    
#     # Split test set into chosen and rejected
#     n_test = len(y_test)
#     test_indices = np.arange(n_test)
#     test_chosen_indices = test_indices[y_test == np.unique(y_test)[1]]  # Assuming chosen scores are higher
#     test_rejected_indices = test_indices[y_test == np.unique(y_test)[0]]
    
#     # Get predictions for chosen and rejected
#     pred_chosen = test_predictions[test_chosen_indices]
#     pred_rejected = test_predictions[test_rejected_indices]
#     actual_chosen = y_test[test_chosen_indices]
#     actual_rejected = y_test[test_rejected_indices]
    
#     # Calculate ranking accuracy
#     ranking_pairs = min(len(pred_chosen), len(pred_rejected))
#     chosen_better = pred_chosen[:ranking_pairs] > pred_rejected[:ranking_pairs]
#     ranking_accuracy = np.mean(chosen_better)
    
#     # Print results
#     print(f"\nResults for {feature_name} features:")
#     print(f"Train R² score: {train_r2:.4f}")
#     print(f"Test R² score: {test_r2:.4f}")
#     print(f"Ranking preservation accuracy: {ranking_accuracy:.4f}")
    
#     return {
#         'train_r2': train_r2,
#         'test_r2': test_r2,
#         'ranking_accuracy': ranking_accuracy,
#         'predictions': test_predictions,
#         'actual': y_test
#     }

# def plot_results(results_768, results_4096, base_dir):
#     """Plot actual vs predicted scores for both feature sets"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     ax1.scatter(results_768['actual'], results_768['predictions'], alpha=0.5)
#     ax1.plot([min(results_768['actual']), max(results_768['actual'])], 
#              [min(results_768['actual']), max(results_768['actual'])], 'r--')
#     ax1.set_title(f"768d Features (R² = {results_768['test_r2']:.4f})")
#     ax1.set_xlabel("Actual Scores")
#     ax1.set_ylabel("Predicted Scores")
    
#     ax2.scatter(results_4096['actual'], results_4096['predictions'], alpha=0.5)
#     ax2.plot([min(results_4096['actual']), max(results_4096['actual'])], 
#              [min(results_4096['actual']), max(results_4096['actual'])], 'r--')
#     ax2.set_title(f"4096d Features (R² = {results_4096['test_r2']:.4f})")
#     ax2.set_xlabel("Actual Scores")
#     ax2.set_ylabel("Predicted Scores")
    
#     plt.suptitle('Feature Dimension Comparison: Actual vs Predicted Scores')
#     plt.tight_layout()
    
#     # Save plot in the same directory as the features
#     plot_path = os.path.join(base_dir, 'feature_comparison_results.png')
#     plt.savefig(plot_path)
#     plt.close()
    
#     print(f"\nPlot saved as: {plot_path}")

# def main():
#     base_dir = os.path.join('features', 'Skywork-Reward-Llama-3.1-8B')
    
#     if not os.path.exists(base_dir):
#         raise FileNotFoundError(f"Directory not found: {base_dir}")
    
#     print(f"Loading data from: {base_dir}")
    
#     (features_chosen_768, features_rejected_768,
#      features_chosen_4096, features_rejected_4096,
#      scores_chosen, scores_rejected) = load_data(base_dir)
    
#     results_768 = evaluate_features(
#         features_chosen_768, features_rejected_768,
#         scores_chosen, scores_rejected,
#         "768-dimensional"
#     )
    
#     results_4096 = evaluate_features(
#         features_chosen_4096, features_rejected_4096,
#         scores_chosen, scores_rejected,
#         "4096-dimensional"
#     )
    

#     ### test random features as well 
#     results_4096 = evaluate_features(
#         features_chosen_4096, features_rejected_4096,
#         scores_chosen, scores_rejected,
#         "random features of shortened size"
#     )
#     plot_results(results_768, results_4096, base_dir)

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         raise




import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

def load_data(base_dir):
    """Load all required data files and create random baseline"""


    args = parse_args()
    features_chosen_path = os.path.join(base_dir, 'features_chosen_full_length_{}.pkl'.format(args.filter_by_subset))
    features_rejected_path = os.path.join(base_dir, 'features_rejected_full_length_{}.pkl'.format(args.filter_by_subset))
    scores_chosen_path = os.path.join(base_dir, 'scores_chosen.pkl')
    scores_rejected_path = os.path.join(base_dir, 'scores_rejected.pkl')
    
    # Load each file
    print("Loading data files...")
    with open(features_chosen_path, 'rb') as f:
        features_chosen_4096 = pickle.load(f).to(torch.float32)
    with open(features_rejected_path, 'rb') as f:
        features_rejected_4096 = pickle.load(f).to(torch.float32)
    with open(scores_chosen_path, 'rb') as f:
        scores_chosen = pickle.load(f).astype(np.float32)
    with open(scores_rejected_path, 'rb') as f:
        scores_rejected = pickle.load(f).astype(np.float32)
    # # Construct full paths
    # features_chosen_path = os.path.join(base_dir, 'features_chosen_full_length.pkl')
    # features_rejected_path = os.path.join(base_dir, 'features_rejected_full_length.pkl')
    # scores_chosen_path = os.path.join(base_dir, 'scores_chosen.pkl')
    # scores_rejected_path = os.path.join(base_dir, 'scores_rejected.pkl')
    
    # # Load each file
    # print("Loading data files...")
    # with open(features_chosen_path, 'rb') as f:
    #     features_chosen_4096 = pickle.load(f)
    # with open(features_rejected_path, 'rb') as f:
    #     features_rejected_4096 = pickle.load(f)
    # with open(scores_chosen_path, 'rb') as f:
    #     scores_chosen = pickle.load(f)
    # with open(scores_rejected_path, 'rb') as f:
    #     scores_rejected = pickle.load(f)
    
    # Extract 768d features
    features_chosen_768 = features_chosen_4096[:, :768]
    features_rejected_768 = features_rejected_4096[:, :768]
    
    # Generate random features with same shape as 768d features
    np.random.seed(42)  # for reproducibility
    random_features_chosen = np.random.randn(*features_chosen_768.shape)
    random_features_rejected = np.random.randn(*features_rejected_768.shape)
    
    print(f"\nFeature shapes:")
    print(f"4096d features - Chosen: {features_chosen_4096.shape}, Rejected: {features_rejected_4096.shape}")
    print(f"768d features - Chosen: {features_chosen_768.shape}, Rejected: {features_rejected_768.shape}")
    print(f"Random features - Chosen: {random_features_chosen.shape}, Rejected: {random_features_rejected.shape}")
    print(f"Scores - Chosen: {len(scores_chosen)}, Rejected: {len(scores_rejected)}")
    
    return (features_chosen_768, features_rejected_768, 
            features_chosen_4096, features_rejected_4096,
            random_features_chosen, random_features_rejected,
            scores_chosen, scores_rejected)

def test_ranking_preservation(model, features_chosen, features_rejected, scores_chosen, scores_rejected):
    num_samples = len(features_chosen)
    indices = np.arange(num_samples)
    pairs = [(features_chosen[i], features_rejected[i], scores_chosen[i], scores_rejected[i]) for i in indices]

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    correct_count = 0
    total_count = 0

    for chosen_feat, rejected_feat, gold_score_chosen, gold_score_rejected in test_pairs:
        with torch.no_grad():
            chosen_score = model.predict([chosen_feat])[0]
            rejected_score = model.predict([rejected_feat])[0]

        # Compare model predictions with gold scores
        is_correct = (chosen_score > rejected_score) == (gold_score_chosen > gold_score_rejected)
        
        # Increment counts
        correct_count += int(is_correct)
        total_count += 1

    # Step 4: Calculate and print Ranking Preservation Accuracy
    ranking_preservation_accuracy = (correct_count / total_count) * 100


    return ranking_preservation_accuracy


def evaluate_features(features_chosen, features_rejected, scores_chosen, scores_rejected, feature_name=""):
    """Evaluate feature set performance with train-test split"""
    
    # Combine features and scores
    X = np.vstack([features_chosen, features_rejected])
    y = np.concatenate([scores_chosen, scores_rejected])
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    # Get predictions
    train_predictions = reg.predict(X_train)
    test_predictions = reg.predict(X_test)
    
    # Calculate R² scores
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # # Split test set into chosen and rejected
    # n_test = len(y_test)
    # test_indices = np.arange(n_test)
    # test_chosen_indices = test_indices[y_test == np.unique(y_test)[1]]
    # test_rejected_indices = test_indices[y_test == np.unique(y_test)[0]]
    
    # # Get predictions for chosen and rejected
    # pred_chosen = test_predictions[test_chosen_indices]
    # pred_rejected = test_predictions[test_rejected_indices]
    
    # # Calculate ranking accuracy
    # ranking_pairs = min(len(pred_chosen), len(pred_rejected))
    # chosen_better = pred_chosen[:ranking_pairs] > pred_rejected[:ranking_pairs]
    # ranking_accuracy = np.mean(chosen_better)

    ranking_accuracy = test_ranking_preservation(reg, features_chosen, features_rejected, scores_chosen, scores_rejected)

    # Print results
    print(f"\nResults for {feature_name} features:")
    print(f"Train R² score: {train_r2:.4f}")
    print(f"Test R² score: {test_r2:.4f}")
    print(f"Ranking preservation accuracy: {ranking_accuracy:.4f}")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'ranking_accuracy': ranking_accuracy,
        'predictions': test_predictions,
        'actual': y_test
    }

def plot_results(results_768, results_4096, results_random, base_dir):
    """Plot actual vs predicted scores for all feature sets"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 768d results
    ax1.scatter(results_768['actual'], results_768['predictions'], alpha=0.5)
    ax1.plot([min(results_768['actual']), max(results_768['actual'])], 
             [min(results_768['actual']), max(results_768['actual'])], 'r--')
    ax1.set_title(f"768d Features\n(R² = {results_768['test_r2']:.4f}, Ranking Acc = {results_768['ranking_accuracy']:.4f})")
    ax1.set_xlabel("Actual Scores")
    ax1.set_ylabel("Predicted Scores")
    
    # Plot 4096d results
    ax2.scatter(results_4096['actual'], results_4096['predictions'], alpha=0.5)
    ax2.plot([min(results_4096['actual']), max(results_4096['actual'])], 
             [min(results_4096['actual']), max(results_4096['actual'])], 'r--')
    ax2.set_title(f"4096d Features\n(R² = {results_4096['test_r2']:.4f}, Ranking Acc = {results_4096['ranking_accuracy']:.4f})")
    ax2.set_xlabel("Actual Scores")
    ax2.set_ylabel("Predicted Scores")
    
    # Plot random feature results
    ax3.scatter(results_random['actual'], results_random['predictions'], alpha=0.5)
    ax3.plot([min(results_random['actual']), max(results_random['actual'])], 
             [min(results_random['actual']), max(results_random['actual'])], 'r--')
    ax3.set_title(f"Random Features\n(R² = {results_random['test_r2']:.4f}, Ranking Acc = {results_random['ranking_accuracy']:.4f})")
    ax3.set_xlabel("Actual Scores")
    ax3.set_ylabel("Predicted Scores")
    
    plt.suptitle('Feature Comparison: Actual vs Predicted Scores')
    plt.tight_layout()
    
    # Save plot in the same directory as the features
    plot_path = os.path.join(base_dir, 'feature_comparison_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nPlot saved as: {plot_path}")

def main():
    # Base directory containing the feature files
    base_dir = os.path.join('.', 'features', 'Skywork-Reward-Llama-3.1-8B')
    
    # Ensure the directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    print(f"Loading data from: {base_dir}")
    
    # Load data
    (features_chosen_768, features_rejected_768,
     features_chosen_4096, features_rejected_4096,
     random_features_chosen, random_features_rejected,
     scores_chosen, scores_rejected) = load_data(base_dir)
    
    # Evaluate 768d features
    results_768 = evaluate_features(
        features_chosen_768, features_rejected_768,
        scores_chosen, scores_rejected,
        "768-dimensional"
    )
    
    # Evaluate 4096d features
    results_4096 = evaluate_features(
        features_chosen_4096, features_rejected_4096,
        scores_chosen, scores_rejected,
        "4096-dimensional"
    )
    
    # Evaluate random features
    results_random = evaluate_features(
        random_features_chosen, random_features_rejected,
        scores_chosen, scores_rejected,
        "random 768-dimensional"
    )
    
    # Plot comparison
    plot_results(results_768, results_4096, results_random, base_dir)
    
    # Print summary comparison
    print("\nSummary Comparison:")
    print("-" * 60)
    print(f"{'Feature Type':<15} {'Test R²':>10} {'Ranking Accuracy':>20}")
    print("-" * 60)
    print(f"{'4096d'::<15} {results_4096['test_r2']:>10.4f} {results_4096['ranking_accuracy']:>20.4f}")
    print(f"{'768d'::<15} {results_768['test_r2']:>10.4f} {results_768['ranking_accuracy']:>20.4f}")
    print(f"{'Random'::<15} {results_random['test_r2']:>10.4f} {results_random['ranking_accuracy']:>20.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise