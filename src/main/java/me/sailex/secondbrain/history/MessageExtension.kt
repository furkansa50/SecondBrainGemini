@file:JvmName("MessageConverter")
package me.sailex.secondbrain.history

import io.github.ollama4j.models.chat.OllamaChatMessage
import io.github.ollama4j.models.chat.OllamaChatMessageRole
import me.sailex.secondbrain.llm.player2.model.Player2ChatMessage
import me.sailex.secondbrain.llm.player2.model.Player2ResponseMessage
import me.sailex.secondbrain.llm.roles.Player2ChatRole
import com.google.cloud.vertexai.api.Content
import com.google.cloud.vertexai.api.Part

// player2
fun Player2ResponseMessage.toMessage(): Message = Message(
    this.content,
    this.role.toString().lowercase()
)

fun Message.toPlayer2ChatMessage(): Player2ChatMessage = Player2ChatMessage(
    Player2ChatRole.valueOf(this.role.uppercase()),
    this.message
)

// ollama
fun OllamaChatMessage.toMessage(): Message = Message(
    this.content,
    this.role.toString().lowercase()
)

fun Message.toOllamaChatMessage(): OllamaChatMessage = OllamaChatMessage(
    OllamaChatMessageRole.getRole(this.role),
    this.message
)

// gemini
fun Message.toGeminiContent(): Content {
    val part = Part.newBuilder().setText(this.message).build()
    // Map roles: "system" and "user" -> "user", "assistant" and "model" -> "model"
    val geminiRole = when (this.role.lowercase()) {
        "system", "user" -> "user"
        "assistant", "model" -> "model"
        else -> "user"
    }
    return Content.newBuilder()
        .setRole(geminiRole)
        .addParts(part)
        .build()
}

fun toGeminiContents(messages: List<Message>): List<Content> {
    // Handle system messages separately as Gemini requires special handling
    val systemMessages = messages.filter { it.role.lowercase() == "system" }
    val otherMessages = messages.filter { it.role.lowercase() != "system" }
    
    // For now, prepend system messages as user messages
    // In production, you might want to use GenerativeModel.Builder.systemInstruction()
    val allMessages = systemMessages + otherMessages
    return allMessages.map { it.toGeminiContent() }
}
