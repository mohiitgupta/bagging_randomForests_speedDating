The code is written using python version 3. The speed dating dataset has been used. The project contains implementation of decision tree, bagged decision trees and random forests. These implementations can be found in `trees.py`. Also, in bonus.py, there is implementation of boosted decision trees which performs the best and gives a test accuracy of 0.77 when tuned with depth limit of 3 and num_trees of 50.

Time taken to run decision tree is around 17 seconds; for bagging its around 8 minutes; for random forests its around 2 minutes. Finally for the bonus question, boosted decision trees, its around 5 minutes.

The preprocessing code is in preprocess_assg4.py. The core logic is found in trees.py. For random forests, the used features have been excluded at each depth. Thus, I sample sqrt(p) features randomly from the unused features at each level.

The code for plotting learning curves showing change in average accuracy of the 3 models with depth_limit is in `cv_depth.py`. The learning curves showing change in accuracy with training data size have been plotted using `cv_frac.py`. Finally, `cv_numtrees.py` contains code for plotting curves for change in number of trees for random forests and bagging.

Training accuracy for decision tree, bagging and random forests are 0.78, 0.79 and 0.78 respectively. The test accuracy scores for these are around 0.70, 0.75, 0.75 respectively.

The boosted decision trees model has been implemented for the bonus question i.e. part 6. Adaboost algorithm has been used and it outperforms decision tree, bagging and random forests. The tuned parameters are depth limit 3 and num_trees = 50. The training accuracy boosted decision tree with these parameters is 0.80 and test accuracy is 0.77.

The report along with learning curves is in CS573_HW4.pdf.

Happy Coding!