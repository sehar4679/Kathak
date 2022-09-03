# this file can be used to convert the confidence of prediction to scores that range from 1-10
# just copy the confidence that is generated after prediction of videos into respective list.
# make sure only the positive prectioon for each class is pasted

import numpy as np

# confidence of pose 1 and pose 2 predictions
pose_1 = [9.9922872e-01,9.9931741e-01,9.9926037e-01,9.9927348e-01,9.9910480e-01,9.9923730e-01,9.9910247e-01,9.9912935e-01,9.9928361e-01,9.9929178e-01,9.9912828e-01,9.9905902e-01,9.9913931e-01,9.9907100e-01,9.9916804e-01,9.9902809e-01,9.9914360e-01,9.9919516e-01,9.9917632e-01,9.9924940e-01]
pose_2 = [9.9435729e-01,9.9400324e-01,9.9398732e-01,9.9519581e-01,9.9445277e-01,9.9514884e-01,9.9520689e-01,9.9435943e-01,9.9481440e-01,9.9397475e-01,9.9551892e-01,9.9489343e-01,9.9484384e-01,9.9518967e-01,9.9533325e-01,9.9407429e-01,9.9475163e-01,9.9459994e-01,9.9498320e-01,9.9603540e-01]


# finding the minimum and maximum of the condifence score to normalize the scores generated
x1 = np.array(pose_1)
max_x1 = np.max(x1)
min_x1 = np.min(x1)
print(max_x1)
print(min_x1)

print('\n\n')

x2 = np.array(pose_2)
max_x2 = np.max(x2)
min_x2 = np.min(x2)
print(max_x2)
print(min_x2)


print('\n\n')


# generating the scores using simple formula.
scale_min1 = 0.9985
sclae_max1 = 0.9995
for i in x1:
    score = (((i-scale_min1)/(sclae_max1-scale_min1))*(10-0))+0
    print(score)
    
print('\n\n')
    

scale_min2 = 0.988
sclae_max2 = 0.998
for i in x2:
    score = (((i-scale_min2)/(sclae_max2-scale_min2))*(10-0))+0
    print(score)