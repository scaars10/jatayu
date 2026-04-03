# Deep Research Feature

## Overview

The Deep Research feature allows Jatayu to perform in-depth research on a given topic. When a user requests deep research, Jatayu kicks off an asynchronous process to gather, synthesize, and report on the topic. This allows the user to continue interacting with Jatayu while the research is being performed in the background. Once the research is complete, Jatayu will send a detailed report to the user.

## How it works

The Deep Research feature is implemented as a tool that can be triggered by the `ChatAgent`. When the `ChatAgent` detects that a user's request requires deep research, it calls the `start_deep_research_task` tool. This tool starts an asynchronous task that runs a dedicated "researcher" agent.

The researcher agent has access to the following tools:
- `WebSearchTool`: To search the web for relevant information.
- `WebFetchTool`: To read the content of web pages.

The researcher agent will iteratively search and read until it has gathered enough information to answer the user's request. It will then synthesize the information into a detailed report, which is sent back to the user via the NATS messaging system.

## How to use

To use the Deep Research feature, simply ask Jatayu to perform deep research on a topic. For example:

> "Can you do a deep research on AI advancements?"

Jatayu will then start the research process and notify you that it has begun. Once the research is complete, you will receive a message with the detailed report.
