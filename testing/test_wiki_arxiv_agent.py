import pytest
from unittest.mock import patch, MagicMock
from LangchainAgents.knowledge import wiki_arxiv_agent


class TestWikiArxivAgent:
    @patch('LangchainAgents.knowledge.AgentExecutor.invoke')
    def test_wiki_arxiv_agent_successful_execution(self, mock_invoke):
        # Mock the return value of the invoke method
        mock_invoke.return_value = "Expected Result"

        # Call the function with a sample query
        result = wiki_arxiv_agent("Sample Query")

        # Assert that the result is as expected
        assert result == "Expected Result"

    @patch('LangchainAgents.knowledge.AgentExecutor.invoke')
    def test_wiki_arxiv_agent_error_handling(self, mock_invoke):
        # Simulate an exception being raised during invocation
        mock_invoke.side_effect = Exception("Simulated Error")

        # Call the function with a sample query
        result = wiki_arxiv_agent("Sample Query")

        # Assert that the result is None due to the exception
        assert result is None

    @patch('LangchainAgents.knowledge.ChatOllama')
    @patch('LangchainAgents.knowledge.ChatPromptTemplate.from_messages')
    def test_wiki_arxiv_agent_integration(self, mock_prompt, mock_llm):
        # Mock the language model and prompt template
        mock_llm.return_value = MagicMock()
        mock_prompt.return_value = MagicMock()

        # Call the function with a sample query
        result = wiki_arxiv_agent("Sample Query")

        # Assert that the language model and prompt template were initialized
        mock_llm.assert_called_once_with(model="llama3.1")
        mock_prompt.assert_called_once()