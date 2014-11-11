from numpy import average, sqrt, mean, square

# Mean Average Error
def MAE(deviation):
    print "The MAE is : " + str(average(deviation))

# Root Mean Square Error
def RMSE(deviation):
    print "The RMSE is : " + str(sqrt(mean(square(deviation))))

