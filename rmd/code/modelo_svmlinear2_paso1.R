X <- entrenamiento[, 1:256]
Y <- entrenamiento$digito
modelo <- train(x = X,
                y = Y,
                method = "svmLinear2")
 
