library(shiny)

# Define the initials and topics
initials <- c("DB", "DT", "PL", "TG", "YR")
topics <- c("Research", "Wiki", "Modelling", "Review/Test", "Project", "EDA")

# Create a function to generate input IDs
make_input_id <- function(topic, initial) {
  paste0(topic, "_", initial)
}

# UI
ui <- fluidPage(
  titlePanel("Group Contribution Calculator"),
  sidebarLayout(
    sidebarPanel(
      lapply(topics, function(topic) {
        lapply(initials, function(initial) {
          numericInput(
            make_input_id(topic, initial),
            paste0(topic, " (", initial, "):"),
            value = 20, min = 0, max = 100, step = 5
          )
        })
      }),
      actionButton("submit", "Submit")
    ),
    mainPanel(
      verbatimTextOutput("summary")
    )
  )
)

# Server
server <- function(input, output) {
  observeEvent(input$submit, {
    # Check if contributions sum up to 100 for each topic
    valid <- vapply(topics, function(topic) {
      contributions <- vapply(initials, function(initial) {
        input[[make_input_id(topic, initial)]]
      }, numeric(1))
      sum(contributions) == 100
    }, logical(1))
    
    if (all(valid)) {
      # Calculate averages for each topic
      averages <- vapply(topics, function(topic) {
        contributions <- vapply(initials, function(initial) {
          input[[make_input_id(topic, initial)]]
        }, numeric(1))
        mean(contributions)
      }, numeric(1))
      
      # Display the summary
      output$summary <- renderPrint({
        cat("Average Contributions:\n")
        mapply(function(topic, avg) {
          cat(topic, ": ", avg, "%\n", sep = "")
        }, topics, averages)
      })
    } else {
      showModal(modalDialog(
        title = "Error",
        "Contributions must sum up to 100% for each topic.",
        easyClose = TRUE
      ))
    }
  })
}

shinyApp(ui, server)





