#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library("shiny")
library("kernlab")
library("ROCR")

# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("Support Vector Machine"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      sidebarPanel(
        sliderInput("number",
                    "Number of observations:",
                    min = 1,
                    max = 100,
                    value = 30, step = 5),
        sliderInput("distance",
                    "Distance between groups:",
                    min = 0.5,
                    max = 5.0,
                    value = 2.0, step = 0.5),
        sliderInput("C",
                    "Cost of constraints violation:",
                    min = 0.1,
                    max = 16.0,
                    value = 1.0, step = 0.1),
        selectInput("kernel","Kernel", 
                    choices = list("Radial Basis kernel" = "rbfdot", "Linear kernel" = "vanilladot"), selected = "rbfdot")
      ),

      # Show a plot of the generated distribution
      mainPanel(
         plotOutput("distPlot", height = "400px"),
         plotOutput("rocPlot", height = "300px")
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

   output$distPlot <- renderPlot({
     
      set.seed(input$number + input$distance)
      df <- data.frame(G = rep(c(0,1), each = input$number),
                      x = c(rnorm(n = input$number, mean = 0, sd = 1), rnorm(n = input$number, mean = 0 + input$distance, sd = 1)),
                      y = c(rnorm(n = input$number, mean = 0, sd = 1), rnorm(n = input$number, mean = 0 + input$distance, sd = 1)))
     

      modelo <- ksvm(G ~ ., 
                     data = df, 
                     type = "C-svc", 
                     C = input$C,
                     kernel = input$kernel)
      
      # table(predict(modelo), data.and$class)
      plot(modelo, 
           data = df)


   })
   
   output$rocPlot <- renderPlot({
     
     set.seed(input$number + input$distance)
     df <- data.frame(G = rep(c(0,1), each = input$number),
                      x = c(rnorm(n = input$number, mean = 0, sd = 1), rnorm(n = input$number, mean = 0 + input$distance, sd = 1)),
                      y = c(rnorm(n = input$number, mean = 0, sd = 1), rnorm(n = input$number, mean = 0 + input$distance, sd = 1)))
     
     
     modelo <- ksvm(G ~ ., 
                    data = df, 
                    type = "C-svc", 
                    C = input$C,
                    kernel = input$kernel)
     
     par(mfrow = c(1,2))
     
     plot(predict(modelo), pch = 16, col = rep(c("red", "blue"), each = input$number),
          xlab = "Observations", ylab = "Class", main = "Classification")
     
     pred <- ROCR::prediction(predictions = predict(modelo), labels = df$G, label.ordering = c(0,1))
     perf <- ROCR::performance(prediction.obj = pred, measure = 'tpr', x.measure = 'fpr')
     plot(perf, main = "ROC Analysis")
     
     
     
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

