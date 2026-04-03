# Interactive Research Feature

## Overview

The Interactive Research feature allows Jatayu to perform in-depth research on a given topic in an interactive, step-by-step way. When a user requests deep research, Jatayu kicks off an asynchronous process to gather, synthesize, and report on the topic. The user can then check the status of the research task, provide feedback, and continue the research from the last step.

## How it works

The Interactive Research feature is built on top of the Deep Research feature. It uses a new `research_tasks` table in the database to store the state of each research task, including the current step of the research process.

The research process is broken down into the following steps:
1.  `gather_sources`: Gather a list of relevant URLs for the research.
2.  `read_sources`: Read the content of the URLs and extract the relevant information.
3.  `synthesize_report`: Synthesize the information into a detailed report.
4.  `wait_for_feedback`: Wait for feedback from the user before continuing to the next step.

When a user requests deep research, a new entry is created in the `research_tasks` table with the status `pending` and the step `gather_sources`. A background task is then started to perform the research. The background task executes the current step, sends a proactive update to the user, and then updates the status and step of the task in the database.

The user can interact with the research task in the following ways:
-   **Receive proactive updates:** Jatayu will automatically send updates as the research progresses.
-   **Check the status:** The `get_research_task_status` tool can be used to check the current status and step of the task.
-   **Provide feedback:** The `provide_feedback_to_research_task` tool can be used to provide feedback at any step of the process. The feedback will be used in the subsequent steps to refine the research.
-   **Continue the research:** The `continue_research_task` tool can be used to continue the research from the last step.

## How to use

To start a deep research task, ask Jatayu to perform deep research on a topic. For example:

> "Can you do a deep research on AI advancements?"

Jatayu will then start the research process and notify you that it has begun, along with a task ID.

To check the status of a research task, you can ask Jatayu for the status of the task, referencing the task ID. For example:

> "what is the status of task 123"

Jatayu will then provide you with the current status and step of the task.

### Providing Feedback

You can also provide feedback to a research task to guide the research process. To provide feedback, you can tell Jatayu to provide feedback to a task, referencing the task ID and providing your feedback. For example:

> "provide feedback to task 123: focus on the ethics of AI"

Jatayu will then use your feedback to refine the research and provide you with a more accurate report.

### Continuing the research

When a research task is waiting for feedback, you can tell Jatayu to continue the research. For example:

> "continue research on task 123"

Jatayu will then confirm that it is continuing the research, and provide you with the current status and next step. For example:

> "Continuing research on 'AI safety'.
>
> Next step: read_sources.
>
> I will now proceed with the research. I will let you know when the next step is complete."

### Reading PDF files

Jatayu can also read the text content of PDF files. To use this feature, simply provide a URL to a PDF file. For example:

> "read the pdf at https://example.com/ai.pdf"

Jatayu will then read the PDF and provide you with a summary of its content.

### Error Handling

If an error occurs during the research process, Jatayu will notify you with a message that includes the error. The research task will be marked as `failed`. You can then try to start the research again.
