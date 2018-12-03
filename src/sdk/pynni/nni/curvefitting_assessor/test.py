 import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]) 
from curvefitting_assessor import CurvefittingAssessor
from assessor import Assessor, AssessResult

New_Assessor = CurvefittingAssessor()

# Test 1
# All Assessor.Good
print (New_Assessor.assess_trial([0.8018166665981213, 0.9276166671514511, 0.94371666431427, 0.9530999948581059, 0.9599833299716314, 0.9638666659593582, 0.9665500009059906, 0.9718000048398971, 0.9731333416700363, 0.9748333434263865, 0.9751]))

New_Assessor.trial_end(True)

print (New_Assessor.assess_trial([0.9412833386659623, 0.9856666787465413, 0.9902166752020518, 0.9932500064373017, 0.9946500051021576, 0.996666669845581, 0.996666669845581, 0.9970833361148834, 0.9976000022888184, 0.998066668510437, 0.9855]))

New_Assessor.trial_end(True)

print (New_Assessor.assess_trial([0.9398166693250338, 0.9840166797240575, 0.9894000099102656, 0.9919666741291682, 0.9947333383560181, 0.9956166708469391, 0.9973000025749207, 0.9972833359241485, 0.9983500015735626, 0.9981833350658417, 0.9888]))

New_Assessor.trial_end(True)

print (New_Assessor.assess_trial(trial_history=[0.0991, 0.3717, 0.5475, 0.6665, 0.7615, 0.8051, 0.8394, 0.8451, 0.8679, 0.8817, 0.8894, 0.8909, 0.9022, 0.9023, 0.9069, 0.9127, 0.9108, 0.9104, 0.913, 0.9228, 0.9235]))
# y_presict = 0.923224352363 AssessResult.Bad 
# y_actual  = 0.9235
New_Assessor.trial_end(True)

print (New_Assessor.assess_trial([0.0976, 0.1032, 0.1135, 0.101, 0.1135, 0.1135, 0.1028, 0.1028, 0.1032, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1135, 0.1028, 0.1135]))
# y_predict = 0.112334281746 AssessResult.Bad
# y_actual  = 0.1135
New_Assessor.trial_end(True)


# Test 2
# All Assessor.Good
print (New_Assessor.assess_trial(trial_job_id=1, trial_history=[0.8018166665981213, 0.9276166671514511, 0.94371666431427, 0.9530999948581059]))
# Good
New_Assessor.trial_end(1, True)

print (New_Assessor.assess_trial(2, [0.9412833386659623, 0.9856666787465413, 0.9902166752020518, 0.9932500064373017]))
# Good
New_Assessor.trial_end(2, True)

print (New_Assessor.assess_trial(3, [0.9398166693250338, 0.9840166797240575, 0.9894000099102656, 0.9919666741291682]))
# y_predict = 0.9598973 AssessResult.Good
# y_truth = 0.9888
New_Assessor.trial_end(3, True)

print (New_Assessor.assess_trial(4, [0.0991, 0.3717, 0.5475, 0.6665, 0.7615]))
# y_predict = 0.9870663 Assessor.Good
# y_actual  = 0.9235
New_Assessor.trial_end(4, True)

print (New_Assessor.assess_trial(5, [0.0976, 0.1032, 0.1135, 0.101, 0.1135]))
# y_predict = 0.11678 Assessor.Bad 
# y_actual = 0.1135
New_Assessor.trial_end(5, True)

# Test 3 Example of Mnist
ret = (New_Assessor.assess_trial(trial_history=[0.0892, 0.0958, 0.1028, 0.0892, 0.1135, 0.1010]))
print (ret)
New_Assessor.trial_end(ret)
# return Good (The first one will always be Good)
# Actual_y  = 0.098000
# predict_y = NaN

ret = (New_Assessor.assess_trial([0.0958, 0.1032, 0.1032, 0.1032, 0.0974, 0.1009]))
print (ret)
New_Assessor.trial_end(ret)
# return Good
# Actual_y  = 0.098000
# Predict_y = 

ret = (New_Assessor.assess_trial([0.1187, 0.9039, 0.9278, 0.9422, 0.9488, 0.9554]))
print (ret)
New_Assessor.trial_end(ret)
# return Good
# Actual_Result = 0.961800
# Predict_y = 

ret = (New_Assessor.assess_trial([0.0530, 0.6593, 0.7481, 0.8081, 0.8511, 0.8656]))
print (ret)
New_Assessor.trial_end(ret)
# return 
# Actual_y  = 0.939600
# Predict_y = 

ret = (New_Assessor.assess_trial([0.0982, 0.1032, 0.0974, 0.0892, 0.1028, 0.1028]))
print (ret)
New_Assessor.trial_end(ret) 
# Actual_y  = 0.098200
# Predict_y = 

ret = (New_Assessor.assess_trial([0.1010, 0.1009, 0.1028, 0.1009, 0.1032, 0.1032]))
print (ret)
New_Assessor.trial_end(ret)
# Actual_y  = 0.113500
# Predict_y = 

ret = (New_Assessor.assess_trial([0.0980, 0.0957, 0.1028, 0.1028, 0.0980, 0.1028]))
print (ret)
New_Assessor.trial_end(ret)
# Actual_y  = 0.113500
# Predict_y = 

ret = (New_Assessor.assess_trial([0.1136, 0.8348, 0.8924, 0.9240, 0.9494, 0.9568]))
print (ret)
New_Assessor.trial_end(ret)
# Actual_y  = 0.984500
# Predict_y = 

ret = (New_Assessor.assess_trial([0.1032, 0.1135, 0.1134, 0.1134, 0.0973, 0.0980]))
print (ret)
New_Assessor.trial_end(ret)
# Actual_y  = 0.113500
# Predict_y = 

ret = (New_Assessor.assess_trial([0.1032, 0.0980, 0.1135, 0.1032, 0.0980, 0.0974]))
print (ret)
New_Assessor.trial_end(ret)
# Actual_y  = 0.102800
# Predict_y = 
'''
