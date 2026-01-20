package me.sailex.secondbrain.llm.gemini;

import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.Content;
import com.google.cloud.vertexai.api.GenerateContentResponse;
import com.google.cloud.vertexai.generativeai.ChatSession;
import com.google.cloud.vertexai.generativeai.GenerativeModel;
import com.google.cloud.vertexai.generativeai.ResponseHandler;
import me.sailex.secondbrain.exception.LLMServiceException;
import me.sailex.secondbrain.history.Message;
import me.sailex.secondbrain.history.MessageConverter;
import me.sailex.secondbrain.llm.LLMClient;

import java.util.List;

public class GeminiClient implements LLMClient {

	private final GenerativeModel model;
	private final int timeout;
	private final String apiKey;

	/**
	 * Constructor for GeminiClient.
	 *
	 * @param model   the model name (e.g., "gemini-pro")
	 * @param apiKey  the Google API key
	 * @param timeout the timeout in seconds
	 */
	public GeminiClient(String model, String apiKey, int timeout) {
		this.apiKey = apiKey;
		this.timeout = timeout;
		
		// Initialize Vertex AI with API key
		VertexAI vertexAi = new VertexAI.Builder()
				.setApiKey(apiKey)
				.build();
		
		// Create the generative model
		this.model = new GenerativeModel(model, vertexAi);
	}

	@Override
	public Message chat(List<Message> messages) {
		try {
			// Convert our Message list to Gemini Content format
			List<Content> contents = MessageConverter.toGeminiContents(messages);
			
			// Generate content
			GenerateContentResponse response = model.generateContent(contents);
			
			// Extract the response text
			String responseText = ResponseHandler.getText(response);
			
			// Return as Message with "model" role (Gemini's equivalent of "assistant")
			return new Message(responseText, "model");
		} catch (Exception e) {
			throw new LLMServiceException("Could not generate Response for prompt: " 
					+ messages.get(messages.size() - 1).getMessage(), e);
		}
	}

	@Override
	public void checkServiceIsReachable() {
		// Basic check - could be enhanced with actual API health check
		if (apiKey == null || apiKey.isEmpty()) {
			throw new LLMServiceException("Gemini API key is not configured");
		}
	}

	@Override
	public void stopService() {
		// Nothing to stop for Gemini client
	}
}
