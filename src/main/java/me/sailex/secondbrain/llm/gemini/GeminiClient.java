package me.sailex.secondbrain.llm.gemini;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import me.sailex.secondbrain.exception.LLMServiceException;
import me.sailex.secondbrain.history.Message;
import me.sailex.secondbrain.llm.LLMClient;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GeminiClient implements LLMClient {

	private final String model;
	private final String apiKey;
	private final int timeout;
	private final HttpClient httpClient;
	private final ObjectMapper objectMapper;
	private static final String API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/";

	/**
	 * Constructor for GeminiClient.
	 *
	 * @param model   the model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
	 * @param apiKey  the Google API key
	 * @param timeout the timeout in seconds
	 */
	public GeminiClient(String model, String apiKey, int timeout) {
		this.model = model;
		this.apiKey = apiKey;
		this.timeout = timeout;
		this.httpClient = HttpClient.newBuilder()
				.connectTimeout(Duration.ofSeconds(timeout))
				.build();
		this.objectMapper = new ObjectMapper();
	}

	@Override
	public Message chat(List<Message> messages) {
		try {
			// Convert messages to Gemini format
			List<Map<String, Object>> geminiContents = new ArrayList<>();
			
			for (Message message : messages) {
				Map<String, Object> content = new HashMap<>();
				
				// Map roles: "system" and "user" -> "user", "assistant" and "model" -> "model"
				String role = message.getRole().toLowerCase();
				if (role.equals("system") || role.equals("user")) {
					content.put("role", "user");
				} else {
					content.put("role", "model");
				}
				
				Map<String, String> part = new HashMap<>();
				part.put("text", message.getMessage());
				content.put("parts", List.of(part));
				
				geminiContents.add(content);
			}
			
			// Build request body
			Map<String, Object> requestBody = new HashMap<>();
			requestBody.put("contents", geminiContents);
			
			String requestJson = objectMapper.writeValueAsString(requestBody);
			
			// Build API URL
			String url = API_BASE_URL + model + ":generateContent?key=" + apiKey;
			
			// Create HTTP request
			HttpRequest request = HttpRequest.newBuilder()
					.uri(URI.create(url))
					.header("Content-Type", "application/json")
					.timeout(Duration.ofSeconds(timeout))
					.POST(HttpRequest.BodyPublishers.ofString(requestJson))
					.build();
			
			// Send request
			HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
			
			if (response.statusCode() != 200) {
				throw new LLMServiceException("Gemini API error: " + response.statusCode() + " - " + response.body());
			}
			
			// Parse response
			JsonNode jsonResponse = objectMapper.readTree(response.body());
			String responseText = jsonResponse
					.path("candidates").get(0)
					.path("content")
					.path("parts").get(0)
					.path("text").asText();
			
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
