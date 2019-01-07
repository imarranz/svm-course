X <- entrenamiento[, 1:256]
Y <- entrenamiento$digito

modelo <- train(x = X,
                y = Y,
                preProcess = c("scale", "center"),
                trControl = trainControl(method = "cv",
                                         p = 0.8, 
                                         number = 5,
                                         repeats = 5),
                method = "svmLinear2",
                tuneLength = 10,
                maximize = TRUE,
                metric = "Accuracy") 

