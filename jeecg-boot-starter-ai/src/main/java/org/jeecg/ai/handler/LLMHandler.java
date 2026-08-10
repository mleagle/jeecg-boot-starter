package org.jeecg.ai.handler;

import com.alibaba.dashscope.aigc.imagesynthesis.ImageSynthesis;
import com.alibaba.dashscope.aigc.imagesynthesis.ImageSynthesisParam;
import com.alibaba.dashscope.aigc.imagesynthesis.ImageSynthesisResult;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversation;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationOutput;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationParam;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationResult;
import com.alibaba.dashscope.common.MultiModalMessage;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.protocol.Protocol;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.*;
import dev.langchain4j.exception.ToolExecutionException;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.community.model.dashscope.QwenChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.service.tool.ToolProviderResult;
import dev.langchain4j.skills.FileSystemSkill;
import dev.langchain4j.skills.FileSystemSkillLoader;
import dev.langchain4j.skills.Skills;
import dev.langchain4j.skills.shell.ShellSkills;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.StringUtils;
import org.jeecg.ai.enums.QwenImageModelEnum;
import org.jeecg.ai.factory.AiModelFactory;
import org.jeecg.ai.factory.AiModelOptions;
import org.jeecg.ai.prop.AiChatProperties;
import org.jeecg.ai.stream.InternalTokenStream;
import org.springframework.util.CollectionUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 大模型聊天工具类
 *
 * @Author: chenrui
 * @Date: 2025/2/18 14:31
 */
@Slf4j
public class LLMHandler {

	private static final String QWEN_IMAGE_3_MODEL_PREFIX = "qwen-image-3";
	private static final int QWEN_IMAGE_3_MAX_OUTPUT_COUNT = 6;
	private static final int QWEN_IMAGE_3_MAX_INPUT_COUNT = 3;
	private static final String DASHSCOPE_COMPATIBLE_MODE_PATH = "/compatible-mode/v1";
	private static final String DASHSCOPE_NATIVE_API_PATH = "/api/v1";
	private static final String DASHSCOPE_MULTIMODAL_SERVICE_PATH = "/services/aigc/multimodal-generation/generation";
	private static final String DASHSCOPE_MULTIMODAL_PATH = "/aigc/multimodal-generation/generation";

    private AiChatProperties aiChatProperties;

    private final ServiceOutputParser serviceOutputParser = new ServiceOutputParser();


    public LLMHandler(AiChatProperties aiChatProperties) {
        this.aiChatProperties = aiChatProperties;
    }

    public LLMHandler() {
    }


    /**
     * 确保aiParams
     *
     * @param params
     * @return
     * @author chenrui
     * @date 2025/3/12 15:22
     */
    private AIParams ensureParams(AIParams params) {
        if (null == params || StringUtils.isEmpty(params.getBaseUrl())) {
            params = getDefaultModel(params);
        }
        if (null == params) {
            throw new IllegalArgumentException("大语言模型参数为空");
        }
        return params;
    }

    /**
     * 获取默认大模型数据
     *
     * @return
     * @author chenrui
     * @date 2025/2/25 19:26
     */
    private AIParams getDefaultModel(AIParams params) {
        if (null == aiChatProperties) {
            log.warn("未配置默认大预言模型");
            return null;
        }
        if (params == null) {
            params = new AIParams();
        }
        params.setProvider(aiChatProperties.getProvider());
        params.setModelName(aiChatProperties.getModel());
        params.setBaseUrl(aiChatProperties.getApiHost());
        params.setApiKey(aiChatProperties.getApiKey());
        params.setSecretKey(aiChatProperties.getCredential().getSecretKey());
        params.setTimeout(aiChatProperties.getTimeout());
        return params;
    }

    public String completions(String message) {
        return completions(Collections.singletonList(UserMessage.from(message)), null);
    }


    /**
     * 问答
     *
     * @param messages
     * @param params
     * @return
     * @author chenrui
     * @date 2025/2/24 17:30
     */
    public String completions(List<ChatMessage> messages, AIParams params) {
        params = ensureParams(params);

        AiModelOptions modelOp = params.toModelOptions();
        ChatModel chatModel = AiModelFactory.createChatModel(modelOp);

        // 整理消息
        CollateMsgResp chatMessage = collateMessage(messages, params);

        // 工具定义
        List<ToolSpecification> toolSpecifications = new ArrayList<>();
        // 工具执行器
        Map<String, ToolExecutor> toolExecutors = new HashMap<>();
        if(null != params.getTools() && !params.getTools().isEmpty()) {
            toolSpecifications = new ArrayList<>(params.getTools().keySet());
            params.getTools().forEach((tool,executor) -> {
                toolExecutors.put(tool.name(),executor);
            });
        }

        // MCP工具解析
        fillMcpTools(params, chatMessage, toolSpecifications, toolExecutors);

        // Skills工具解析（activate_skill 模式：模型通过 FunctionCall 自主激活 skill）
        fillSkillTools(params, chatMessage, toolSpecifications, toolExecutors);

        // Shell Skills工具解析（命令行模式：注册 run_shell_command 工具）
        fillSkillToolsShellMode(params, chatMessage, toolSpecifications, toolExecutors);

        //update-begin---wangshuai---date:20260413  for：[issue/1560]/[issues/9527]AI应用调用千问qwen-plus 大模型 提示messages and prompt must not all null #18-----------
        // 快照当前轮的 UserMessage，供 sanitize 在 chatMemory 淘汰 User 后回填
        UserMessage currentTurnUserMessage = null;
        List<ChatMessage> initialMsgs = chatMessage.chatMemory.messages();
        for (int i = initialMsgs.size() - 1; i >= 0; i--) {
            if (initialMsgs.get(i) instanceof UserMessage) {
                currentTurnUserMessage = (UserMessage) initialMsgs.get(i);
                break;
            }
        }
        //update-end---wangshuai---date:20260413  for：[issue/1560]/[issues/9527]AI应用调用千问qwen-plus 大模型 提示messages and prompt must not all null #18-----------

        String resp = "";
        log.info("[LLMHandler] send message to AI server. message: {}", chatMessage);
        while (true) {
            ChatRequest.Builder requestBuilder = ChatRequest.builder()
                    //update-begin---wangshuai---date:20260413  for：[issue/1560]/[issues/9527]AI应用调用千问qwen-plus 大模型 提示messages and prompt must not all null #18-----------
                    .messages(org.jeecg.ai.stream.InternalTokenStream.reinjectUserMessageIfEvicted(
                            chatMessage.chatMemory.messages(), currentTurnUserMessage));
                    //update-end---wangshuai---date:20260413  for：[issue/1560]/[issues/9527]AI应用调用千问qwen-plus 大模型 提示messages and prompt must not all null #18-----------

            // 判断模型是否支持工具调用
            if(isSupportTools(chatModel.defaultRequestParameters())) {
                requestBuilder = requestBuilder.toolSpecifications(toolSpecifications);
            }
            //update-begin---author:wangshuai---date:2025-12-18---for: 工具调用失败后打印日志，不影响整个流程---
            ChatResponse response = null;
            try {
                response = chatModel.chat(requestBuilder.build());
            } catch (ToolExecutionException e) {
                log.error("工具调用失败：{}", e.getMessage(), e);
                break;
            }
            //update-end---author:wangshuai---date:2025-12-18---for: 工具调用失败后打印日志，不影响整个流程---
            AiMessage aiMessage = response.aiMessage();
            // 部分 API 不接受 content 为 null 的 AiMessage，对含工具调用但无文本的消息补空字符串
            if (aiMessage.hasToolExecutionRequests() && aiMessage.text() == null) {
                aiMessage = AiMessage.from("", aiMessage.toolExecutionRequests());
            }
            chatMessage.chatMemory.add(aiMessage);

            // 没有工具调用，则解析文本并结束
            if (aiMessage.toolExecutionRequests() == null || aiMessage.toolExecutionRequests().isEmpty()) {
                resp = (String) serviceOutputParser.parse(response, String.class);
                break;
            }

            // 有工具调用：逐个执行并将结果以 ToolExecutionResultMessage 追加到历史
            for (ToolExecutionRequest toolExecReq : aiMessage.toolExecutionRequests()) {
                ToolExecutor executor = toolExecutors.get(toolExecReq.name());
                if (executor == null) {
                    throw new IllegalStateException("未找到工具执行器: " + toolExecReq.name());
                }
                log.info("[LLMHandler] Executing tool: {} ", toolExecReq.name());
                try{
                    //update-begin---author:wangshuai---date:2026-03-17---for:【QQYUN-14935】构建skills插件---
                    // 1. 构建调用上下文，携带会话ID和当前时间戳
                    InvocationContext ctx = InvocationContext.builder()
                            .chatMemoryId(chatMessage.chatMemory.id())
                            .timestampNow()
                            .build();
                    // 2. 执行工具并获取结果文本
                    String result = executor.executeWithContext(toolExecReq, ctx).resultText();
                    //update-end---author:wangshuai---date:2026-03-17---for:【QQYUN-14935】构建skills插件---
                    // 防止工具返回 null 导致 API 报错 "expected a string, got null"
                    if (result == null) {
                        result = "";
                    }
                    ToolExecutionResultMessage resultMsg = ToolExecutionResultMessage.from(toolExecReq, result);
                    chatMessage.chatMemory.add(resultMsg);
                }catch (ToolExecutionException e){
                    log.info("插件运行失败，原因：{}",e.getMessage(),e);
                }
   
            }
        }


        log.info("[LLMHandler] Received the AI's response . message: {}", resp);
        return resp;
    }

    /**
     * 模型是否支持工具调用
     * @param parameters
     * @return
     * @author chenrui
     * @date 2025/10/14 16:27
     */
    private boolean isSupportTools(ChatRequestParameters parameters) {
        String modelName = parameters.modelName();
        boolean isMultimodalModel = modelName.contains("-vl-") || modelName.contains("-audio-") || modelName.contains("-omni-");
        // 通过QwenChatRequestParameters的isMultimodalModel标记判断（如qwen3.5-plus配置了image_url）
        if (!isMultimodalModel && parameters instanceof QwenChatRequestParameters qwenParams) {
            isMultimodalModel = Boolean.TRUE.equals(qwenParams.isMultimodalModel());
        }
        // 多模态模型不支持工具调用
        return !isMultimodalModel;
    }

    /**
     * 聊天(流式)
     *
     * @param messages
     * @param params
     * @return
     * @author chenrui
     * @date 2025/2/24 17:29
     */
    public TokenStream chat(List<ChatMessage> messages, AIParams params) {
        params = ensureParams(params);
        if (null == params) {
            throw new IllegalArgumentException("大语言模型参数为空");
        }

        // model
        AiModelOptions modelOp = params.toModelOptions();
        StreamingChatModel streamingChatModel = AiModelFactory.createStreamingChatModel(modelOp);

        // 工具定义
        List<ToolSpecification> toolSpecifications = new ArrayList<>();
        // 工具执行器
        Map<String, ToolExecutor> toolExecutors = new HashMap<>();
        if(null != params.getTools() && !params.getTools().isEmpty()) {
            toolSpecifications = new ArrayList<>(params.getTools().keySet());
            params.getTools().forEach((tool,executor) -> {
                toolExecutors.put(tool.name(),executor);
            });
        }

        CollateMsgResp chatMessage = collateMessage(messages, params);

        // MCP工具解析
        fillMcpTools(params, chatMessage, toolSpecifications, toolExecutors);

        // Skills工具解析（activate_skill 模式：模型通过 FunctionCall 自主激活 skill）
        fillSkillTools(params, chatMessage, toolSpecifications, toolExecutors);

        // Shell Skills工具解析（命令行模式：注册 run_shell_command 工具）
        fillSkillToolsShellMode(params, chatMessage, toolSpecifications, toolExecutors);

        // 判断模型是否支持工具调用（多模态模型如qwen-vl-ocr不支持）
        if(!isSupportTools(streamingChatModel.defaultRequestParameters())) {
            toolSpecifications = new ArrayList<>();
        }

        log.info("[LLMHandler] send streaming message to AI server. message: {}", chatMessage);
        //update-begin---author:wangshuai---date:2025-12-18---for:【QQYUN-14048】【AI】langchain4j 升级到1.9.1---
        return new InternalTokenStream(
                streamingChatModel,
                toolSpecifications,
                toolExecutors,
                chatMessage.chatMemory,
                chatMessage.augmentationResult != null ? chatMessage.augmentationResult.contents() : null
        );
        //update-end---author:wangshuai---date:2025-12-18---for:【QQYUN-14048】【AI】langchain4j 升级到1.9.1---
    }


    /**
     * 将 MCP ToolProviders 解析并填充到工具规格与执行器集合中
     * @param params AI参数
     * @param chatMessage 整理后的消息对象
     * @param toolSpecifications 现有工具规格集合(将追加)
     * @param toolExecutors 现有工具执行器集合(将追加)
     *
     * @param params
     * @param chatMessage
     * @param toolSpecifications
     * @param toolExecutors
     * @author chenrui
     * @date 2025/10/21 17:34
     */
    private void fillMcpTools(AIParams params,
                              CollateMsgResp chatMessage,
                              List<ToolSpecification> toolSpecifications,
                              Map<String, ToolExecutor> toolExecutors) {
        if (params.getMcpToolProviders() == null || params.getMcpToolProviders().isEmpty()) {
            return;
        }
        for (McpToolProvider toolProvider : params.getMcpToolProviders()) {
            //update-begin---author:wangshuai---date:2025-12-18---for:【QQYUN-14048】【AI】langchain4j 升级到1.9.1---
            ToolProviderRequest request = new ToolProviderRequest(chatMessage.chatMemory.id(),chatMessage.userMessage);
            ToolProviderResult result = toolProvider.provideTools(request);
            if (result != null && result.tools() != null) {
                for (Map.Entry<ToolSpecification, ToolExecutor> entry : result.tools().entrySet()) {
                    toolSpecifications.add(entry.getKey());
                    toolExecutors.put(entry.getKey().name(), entry.getValue());
            //update-end---author:wangshuai---date:2025-12-18---for:【QQYUN-14048】【AI】langchain4j 升级到1.9.1---
                }
            }
        }
    }

    /**
     * 将 Skills 解析并填充到工具规格与执行器集合中
     * 自动判断模式：有现有工具时使用 Tool Mode，否则使用 Shell Mode
     *
     * @param params             AI参数
     * @param chatMessage        整理后的消息对象
     * @param toolSpecifications 现有工具规格集合(将追加)
     * @param toolExecutors      现有工具执行器集合(将追加)
     * @author wangshuai
     * @date 2026/3/16 18:56
     */
    private void fillSkillTools(AIParams params,
                                CollateMsgResp chatMessage,
                                List<ToolSpecification> toolSpecifications,
                                Map<String, ToolExecutor> toolExecutors) {
        if (StringUtils.isEmpty(params.getSkillsDir())) {
            return;
        }

        List<FileSystemSkill> fileSystemSkills;
        try {
            Path skillsPath = Paths.get(params.getSkillsDir());
            fileSystemSkills = FileSystemSkillLoader.loadSkills(skillsPath);
        } catch (Exception e) {
            log.error("从文件系统加载Skills失败: {}", e.getMessage(), e);
            return;
        }

        if (fileSystemSkills.isEmpty()) {
            return;
        }

        log.info("[LLMHandler] Skills loaded: {} from {}", fileSystemSkills.size(), params.getSkillsDir());

        // 注册 Skills 提供的全部工具（read_skill_resource、activate_skill 等）
        // 由模型通过 activate_skill 工具自主决定激活哪个 skill
        Skills skills = Skills.from(fileSystemSkills);
        ToolProviderRequest request = new ToolProviderRequest(chatMessage.chatMemory.id(), chatMessage.userMessage);
        ToolProviderResult result = skills.toolProvider().provideTools(request);
        if (result != null && result.tools() != null) {
            for (Map.Entry<ToolSpecification, ToolExecutor> entry : result.tools().entrySet()) {
                toolSpecifications.add(entry.getKey());
                toolExecutors.put(entry.getKey().name(), entry.getValue());
                log.info("[LLMHandler] Skill tool registered: {}", entry.getKey().name());
            }
        }

        // 将可用 Skills 列表注入系统消息，让模型知道有哪些 skill 可用以及何时调用
        // 解决 qwen-plus 等模型不主动调用 activate_skill 的问题
        String availableSkillsInfo = skills.formatAvailableSkills();
        String skillsPrompt = "\n\n## 可用技能(Skills)\n" +
                "你拥有以下技能。当用户的请求匹配到某个技能的描述或触发词时，" +
                "你必须先调用 `activate_skill` 工具（参数为技能名称）获取详细指令，然后按指令执行。" +
                "不要跳过 activate_skill 直接回答。\n" +
                availableSkillsInfo;
        // 注入运行时上下文（Token、API地址等），供Skill引用
        if (StringUtils.isNotEmpty(params.getSkillsContext())) {
            skillsPrompt += "\n\n## Skills运行时上下文\n" + params.getSkillsContext();
        }
        injectSkillsPromptToSystemMessage(chatMessage.chatMemory, skillsPrompt);
    }

    /**
     * Shell 模式加载 Skills：使用 ShellSkills 将 skill 内容直接注入系统消息，
     * 并注册 run_shell_command + read_skill_resource 工具，不依赖 activate_skill FunctionCall。
     * 适用于命令行场景或不支持 FunctionCall 的模型。
     *
     * @param params             AI参数（需设置 shellSkillsDir）
     * @param chatMessage        整理后的消息对象
     * @param toolSpecifications 现有工具规格集合(将追加)
     * @param toolExecutors      现有工具执行器集合(将追加)
     * @author wangshuai
     * @date 2026/3/18 21:00
     */
    private void fillSkillToolsShellMode(AIParams params,
                                         CollateMsgResp chatMessage,
                                         List<ToolSpecification> toolSpecifications,
                                         Map<String, ToolExecutor> toolExecutors) {
        if (StringUtils.isEmpty(params.getSkillsShellDir())) {
            return;
        }

        List<FileSystemSkill> fileSystemSkills;
        try {
            Path skillsPath = Paths.get(params.getSkillsShellDir());
            fileSystemSkills = FileSystemSkillLoader.loadSkills(skillsPath);
        } catch (Exception e) {
            log.error("从文件系统加载Shell Skills失败: {}", e.getMessage(), e);
            return;
        }

        if (fileSystemSkills.isEmpty()) {
            return;
        }

        log.info("[LLMHandler] Skills loaded (Shell Mode): {} from {}", fileSystemSkills.size(), params.getSkillsShellDir());

        // 使用 ShellSkills：注册 run_shell_command 工具（替代 activate_skill FunctionCall）
        ShellSkills shellSkills = ShellSkills.from(fileSystemSkills);
        ToolProviderRequest request = new ToolProviderRequest(chatMessage.chatMemory.id(), chatMessage.userMessage);
        ToolProviderResult result = shellSkills.toolProvider().provideTools(request);
        if (result != null && result.tools() != null) {
            for (Map.Entry<ToolSpecification, ToolExecutor> entry : result.tools().entrySet()) {
                toolSpecifications.add(entry.getKey());
                toolExecutors.put(entry.getKey().name(), entry.getValue());
                log.info("[LLMHandler] Shell skill tool registered: {}", entry.getKey().name());
            }
        }

        // 将可用 Skills 列表注入系统消息
        String availableSkillsInfo = shellSkills.formatAvailableSkills();
        String skillsPrompt = "\n\n## 可用技能(Skills)\n" +
                "你拥有以下技能。当用户的请求匹配到某个技能的描述或触发词时，" +
                "请按照技能的指令严格执行，直接产出结果。\n" +
                availableSkillsInfo;
        // 注入运行时上下文（Token、API地址等），供Skill引用
        if (StringUtils.isNotEmpty(params.getSkillsContext())) {
            skillsPrompt += "\n\n## Skills运行时上下文\n" + params.getSkillsContext();
        }
        injectSkillsPromptToSystemMessage(chatMessage.chatMemory, skillsPrompt);
    }

    /**
     * 将 Skills 提示注入到 ChatMemory 的系统消息中
     * 如果已有系统消息则追加，否则新增一条系统消息
     *
     * @param chatMemory    消息缓存
     * @param skillsPrompt  Skills 提示文本
     * @author wangshuai
     * @date 2026/3/18 20:00
     */
    private void injectSkillsPromptToSystemMessage(ChatMemory chatMemory, String skillsPrompt) {
        List<ChatMessage> currentMessages = new ArrayList<>(chatMemory.messages());
        boolean hasSystemMessage = false;
        for (int i = 0; i < currentMessages.size(); i++) {
            if (currentMessages.get(i).type() == ChatMessageType.SYSTEM) {
                SystemMessage existingSystem = (SystemMessage) currentMessages.get(i);
                currentMessages.set(i, SystemMessage.from(existingSystem.text() + skillsPrompt));
                hasSystemMessage = true;
                break;
            }
        }
        if (!hasSystemMessage) {
            currentMessages.add(0, SystemMessage.from(skillsPrompt));
        }
        // 重建 chatMemory
        chatMemory.clear();
        currentMessages.forEach(chatMemory::add);
    }

    /**
     * 整理消息
     *
     * @param messages
     * @param params
     * @return
     * @author chenrui
     * @date 2025/3/18 16:52
     */
    private CollateMsgResp collateMessage(List<ChatMessage> messages, AIParams params) {
        if (null == params) {
            params = new AIParams();
        }

        // 获取用户消息
        List<ChatMessage> messagesCopy = new ArrayList<>(messages);
        UserMessage userMessage = null;
        ChatMessage lastMessage = messagesCopy.get(messagesCopy.size() - 1);
        if (lastMessage.type().equals(ChatMessageType.USER)) {
            userMessage = (UserMessage) messagesCopy.remove(messagesCopy.size() - 1);
        } else {
            throw new IllegalArgumentException("最后一条消息必须是用户消息");
        }
        // 从extraParams中提取image_url，构建多模态消息（图片+文本）
        userMessage = appendImageContent(userMessage, params);

        // 最大消息数: 系统消息和用户消息也会计数,所以最大消息增加两个
        int maxMsgNumber = 4 + 2;
        if (null != params.getMaxMsgNumber()) {
            maxMsgNumber = params.getMaxMsgNumber() + 2;
        }


        // 系统消息
        AtomicReference<SystemMessage> systemMessageAto = new AtomicReference<>();
        messagesCopy.removeIf(tempMsg -> {
            if (ChatMessageType.SYSTEM.equals(tempMsg.type())) {
                if (systemMessageAto.get() == null) {
                    systemMessageAto.set((SystemMessage) tempMsg);
                } else {
                    SystemMessage systemMessage = systemMessageAto.get();
                    String text = systemMessage.text() + "\n" + ((SystemMessage) tempMsg).text();
                    systemMessageAto.set(SystemMessage.from(text));
                }
                return true;
            }
            return false;
        });

        // 消息缓存
        ChatMemory chatMemory = MessageWindowChatMemory.builder().maxMessages(maxMsgNumber).build();

        // 添加系统消息到
        if (null != systemMessageAto.get()) {
            chatMemory.add(systemMessageAto.get());
        }
        // 添加历史消息
        messagesCopy.forEach(chatMemory::add);


        // RAG
        AugmentationResult augmentationResult = null;
        if (null != params.getQueryRouter()) {
            DefaultRetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder().queryRouter(params.getQueryRouter()).build();
            if (retrievalAugmentor != null) {
                StringBuilder userQuestion = new StringBuilder();
                List<dev.langchain4j.data.message.Content> contents = new ArrayList<>(userMessage.contents());
                for (int i = contents.size() - 1; i >= 0; i--) {
                    if (contents.get(i) instanceof TextContent) {
                        userQuestion.append(((TextContent) contents.remove(i)).text());
                        userQuestion.append("\n");
                    }
                }
                UserMessage textUserMessage = UserMessage.from(userQuestion.toString());
                Metadata metadata = Metadata.from(textUserMessage, "default", chatMemory.messages());
                AugmentationRequest augmentationRequest = new AugmentationRequest(textUserMessage, metadata);
                augmentationResult = retrievalAugmentor.augment(augmentationRequest);
                textUserMessage = (UserMessage) augmentationResult.chatMessage();
                contents.add(dev.langchain4j.data.message.TextContent.from(textUserMessage.singleText()));
                userMessage = UserMessage.from(contents);
            }
        }
        // 用户消息
        chatMemory.add(userMessage);
        return new CollateMsgResp(chatMemory, augmentationResult, userMessage);
    }

    /**
     * 从extraParams中提取image_url，将图片追加到用户消息中构建多模态消息
     * extraParams格式: {"image_url": {"url": "https://..."}}
     * 如果对话历史中已存在图片内容，则仅移除extraParams中的image_url，不再重复追加
     */
    private UserMessage appendImageContent(UserMessage userMessage, AIParams params) {
        if (params == null || params.getExtraParams() == null || !params.getExtraParams().containsKey("image_url")) {
            return userMessage;
        }
        try {
            //update-begin---author:wangshuai---date:2026-03-30---for:对话中已存在图片时，移除extraParams中的image_url避免重复---
            // 检查当前用户消息是否已包含图片内容（如用户通过聊天界面上传了图片）
            boolean hasImageInContent = hasImageInContent(userMessage);
            if (hasImageInContent) {
                // 消息中已有图片，仅移除extraParams中的image_url，不再重复追加
                params.getExtraParams().remove("image_url");
                return userMessage;
            }
            //update-end---author:wangshuai---date:2026-03-30---for:对话中已存在图片时，移除extraParams中的image_url避免重复---
            Object imageUrlObj = params.getExtraParams().get("image_url");
            String url = null;
            if (imageUrlObj instanceof Map) {
                url = (String) ((Map<?, ?>) imageUrlObj).get("url");
            } else if (imageUrlObj instanceof String) {
                url = (String) imageUrlObj;
            }
            if (url != null && !url.isEmpty()) {
                List<Content> contents = new ArrayList<>(userMessage.contents());
                contents.add(0, ImageContent.from(url));
                // 从extraParams中移除image_url，避免重复传递到请求参数
                params.getExtraParams().remove("image_url");
                return UserMessage.from(contents);
            }
        } catch (Exception e) {
            log.warn("解析extraParams中的image_url失败: {}", e.getMessage());
        }
        return userMessage;
    }

    /**
     * 检查当前对话是否包含图片内容
     */
    private boolean hasImageInContent(ChatMessage userMessage) {
        if (userMessage == null) {
            return false;
        }
        if (userMessage instanceof UserMessage um) {
            for (Content content : um.contents()) {
                if (content instanceof ImageContent) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 消息整理返回值
     * @author chenrui
     * @date 2025/3/18 16:53
     */
    private static class CollateMsgResp {
        public final ChatMemory chatMemory;
        public final AugmentationResult augmentationResult;
        public final UserMessage userMessage;

        public CollateMsgResp(ChatMemory chatMemory, AugmentationResult augmentationResult, UserMessage userMessage) {
            this.chatMemory = chatMemory;
            this.augmentationResult = augmentationResult;
            this.userMessage = userMessage;
        }

        @Override
        public String toString() {
            // 返回完整的消息内容
            return "{messages=" + (chatMemory != null ? chatMemory.messages() : "null") + "}";
        }
    }

    /**
     * 
     * 图像生成
     * 
     * @param prompt
     * @param params
     * @return
     */
    public List<Map<String,Object>> imageGenerate(String prompt, AIParams params) {
        params = ensureParams(params);
        if (null == params) {
            throw new IllegalArgumentException("大语言模型参数为空");
        }

        AiModelOptions options = params.toModelOptions();
		//update-begin---author:scott ---date:20260807  for：【无号】Qwen Image 3按供应商路由多模态接口-----------
		if (isQwenImage3Model(options.getProvider(), options.getModelName())) {
			return imageGenerateByQwenMultimodal(prompt, options, params.imageCount);
		}
		//update-end---author:scott ---date:20260807  for：【无号】Qwen Image 3按供应商路由多模态接口-----------

        ImageModel imageModel = AiModelFactory.createImageModel(options);
        List<Map<String,Object>> result = new ArrayList<>();
        Integer imageCount = params.imageCount;
        int count = (imageCount == null || imageCount < 1) ? 1 : imageCount;
        try {
            for (int i = 0; i < count; i++) {
                Response<Image> resp = imageModel.generate(prompt);
                Image image = resp.content();
                Map<String, Object> item = new HashMap<>();
                if (StringUtils.isNotEmpty(image.base64Data())) {
                    item.put("type", "base64");
                    String base64 = image.base64Data();
                    if (!base64.startsWith("data:")) {
                        base64 = "data:image/png;base64," + base64;
                    }
                    item.put("value", base64);
                } else if (image.url() != null) {
                    item.put("type", "http");
                    item.put("value", image.url().toString());
                }
                result.add(item);
            }
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }

        return result;
    }

	/**
	 * 判断是否为需要走多模态生成接口的 Qwen Image 3 模型。
	 *
	 * @param provider 供应商
	 * @param modelName 模型名称
	 * @return 是否为 Qwen Image 3 模型
	 * @author scott
	 * @since 2026-08-07 【无号】限定Qwen Image 3多模态路由范围
	 */
	static boolean isQwenImage3Model(String provider, String modelName) {
		if (!AiModelFactory.AIMODEL_TYPE_QWEN.equalsIgnoreCase(provider) || StringUtils.isEmpty(modelName)) {
			return false;
		}
		String normalizedModelName = modelName.toLowerCase(Locale.ROOT);
		return QWEN_IMAGE_3_MODEL_PREFIX.equals(normalizedModelName)
				|| normalizedModelName.startsWith(QWEN_IMAGE_3_MODEL_PREFIX + ".");
	}

	//update-begin---author:claude ---date:2026-08-07  for：图片模型测试连接慢，提供测试专用的最小尺寸/最低质量参数，缩短测试耗时-----------
	/**
	 * 获取图片模型【测试连接】专用的最小尺寸，加快测试速度。
	 * 返回 null 表示保持供应商默认尺寸（默认已是最小或无法安全推断）。
	 *
	 * @param provider 供应商
	 * @param modelName 模型名称
	 * @return 测试用最小尺寸
	 */
	public static String resolveTestImageSize(String provider, String modelName) {
		if (StringUtils.isEmpty(provider)) {
			return null;
		}
		String model = modelName == null ? "" : modelName.toLowerCase(Locale.ROOT);
		if (AiModelFactory.AIMODEL_TYPE_OPENAI.equalsIgnoreCase(provider)) {
			// dall-e-2 最小支持 256x256；dall-e-3 / gpt-image 系列最小为 1024x1024
			if (model.startsWith("dall-e-2")) {
				return "256x256";
			}
			return "1024x1024";
		}
		if (AiModelFactory.AIMODEL_TYPE_QWEN.equalsIgnoreCase(provider)) {
			// qwen-image-3 走多模态接口，尺寸为固定支持列表，保持默认
			if (isQwenImage3Model(provider, modelName)) {
				return null;
			}
			// 万相 turbo 系列最小支持 512*512，其余保持默认 1024*1024
			if (model.contains("turbo")) {
				return "512*512";
			}
		}
		return null;
	}

	/**
	 * 获取图片模型【测试连接】专用的最低质量，加快测试速度。
	 * 返回 null 表示无需传递质量参数。
	 *
	 * @param provider 供应商
	 * @param modelName 模型名称
	 * @return 测试用最低质量
	 */
	public static String resolveTestImageQuality(String provider, String modelName) {
		// gpt-image 系列 low 质量可大幅缩短生成耗时（默认auto为高质量，可达60秒以上）
		String model = modelName == null ? "" : modelName.toLowerCase(Locale.ROOT);
		if (AiModelFactory.AIMODEL_TYPE_OPENAI.equalsIgnoreCase(provider) && model.startsWith("gpt-image")) {
			return "low";
		}
		return null;
	}
	//update-end---author:claude ---date:2026-08-07  for：图片模型测试连接慢，提供测试专用的最小尺寸/最低质量参数，缩短测试耗时-----------

	/**
	 * 使用 DashScope 多模态接口完成 Qwen Image 3 文生图。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】支持Qwen Image 3文生图
	 */
	private List<Map<String, Object>> imageGenerateByQwenMultimodal(String prompt, AiModelOptions options, Integer imageCount) {
		List<Map<String, Object>> content = new ArrayList<>();
		content.add(Collections.singletonMap("text", prompt));
		return callQwenImageMultimodal(content, options, imageCount);
	}

	/**
	 * 使用 DashScope 多模态接口完成 Qwen Image 3 图像编辑。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】支持Qwen Image 3图像编辑
	 */
	private List<Map<String, Object>> imageEditByQwenMultimodal(String prompt, List<String> originalImages,
			AiModelOptions options, Integer imageCount) {
		if (originalImages.size() > QWEN_IMAGE_3_MAX_INPUT_COUNT) {
			throw new IllegalArgumentException("Qwen Image 3 最多支持3张输入图片");
		}

		List<Map<String, Object>> content = new ArrayList<>();
		for (String originalImage : originalImages) {
			content.add(Collections.singletonMap("image", normalizeQwenImageData(originalImage)));
		}
		content.add(Collections.singletonMap("text", prompt));
		return callQwenImageMultimodal(content, options, imageCount);
	}

	/**
	 * 调用 Qwen Image 3 多模态生成接口并转换为现有图片返回协议。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】支持Qwen Image 3多模态生成
	 */
	private List<Map<String, Object>> callQwenImageMultimodal(List<Map<String, Object>> content,
			AiModelOptions options, Integer imageCount) {
		AiModelFactory.assertNotEmpty("apiKey不能为空", options.getApiKey());
		int count = normalizeQwenImageCount(imageCount);
		MultiModalMessage userMessage = MultiModalMessage.builder()
				.role(Role.USER.getValue())
				.content(content)
				.build();
		MultiModalConversationParam.MultiModalConversationParamBuilder<?, ?> paramBuilder = MultiModalConversationParam.builder()
				.apiKey(options.getApiKey())
				.model(options.getModelName())
				.messages(Collections.singletonList(userMessage))
				.n(count)
				.watermark(false);
		if (StringUtils.isNotEmpty(options.getImageSize())) {
			paramBuilder.size(options.getImageSize());
		}

		try {
			MultiModalConversation conversation = new MultiModalConversation(
					Protocol.HTTP.getValue(), normalizeDashScopeBaseUrl(options.getBaseUrl()));
			MultiModalConversationResult response = conversation.call(paramBuilder.build());
			return parseQwenImageResponse(response);
		} catch (Exception e) {
			throw new RuntimeException("Qwen Image 3 generation failed: " + e.getMessage(), e);
		}
	}

	/**
	 * 将 Qwen Image 3 响应中的图片地址转换为现有 type/value 结构。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】转换Qwen Image 3响应
	 */
	private List<Map<String, Object>> parseQwenImageResponse(MultiModalConversationResult response) {
		List<Map<String, Object>> result = new ArrayList<>();
		if (response != null && response.getOutput() != null && response.getOutput().getChoices() != null) {
			for (MultiModalConversationOutput.Choice choice : response.getOutput().getChoices()) {
				if (choice.getMessage() == null || choice.getMessage().getContent() == null) {
					continue;
				}
				for (Map<String, Object> item : choice.getMessage().getContent()) {
					Object image = item.get("image");
					if (image != null) {
						result.add(toImageResult(image.toString()));
					}
				}
			}
		}
		if (result.isEmpty()) {
			String errorMessage = response == null ? "empty response" : response.getCode() + ": " + response.getMessage();
			throw new IllegalStateException("Qwen Image 3 response contains no image, " + errorMessage);
		}
		return result;
	}

	/**
	 * 转换单张图片为现有返回协议。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】统一图片返回结构
	 */
	private Map<String, Object> toImageResult(String image) {
		Map<String, Object> result = new HashMap<>();
		if (image.startsWith("http://") || image.startsWith("https://")) {
			result.put("type", "http");
			result.put("value", image);
			return result;
		}

		result.put("type", "base64");
		result.put("value", image.startsWith("data:") ? image : "data:image/png;base64," + image);
		return result;
	}

	/**
	 * 补充 DashScope 接受的 Base64 Data URL 前缀。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】规范Qwen Image 3输入图片
	 */
	private String normalizeQwenImageData(String image) {
		return image.startsWith("data:") || image.startsWith("http://") || image.startsWith("https://")
				? image : "data:image/png;base64," + image;
	}

	/**
	 * DashScope SDK 会自行拼接服务路径，将配置地址归一化为原生 API 基础地址。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】避免多模态接口路径重复拼接
	 */
	private String normalizeDashScopeBaseUrl(String baseUrl) {
		String normalized = baseUrl;
		while (normalized.endsWith("/")) {
			normalized = normalized.substring(0, normalized.length() - 1);
		}
		if (normalized.endsWith(DASHSCOPE_MULTIMODAL_SERVICE_PATH)) {
			normalized = normalized.substring(0, normalized.length() - DASHSCOPE_MULTIMODAL_SERVICE_PATH.length());
		} else if (normalized.endsWith(DASHSCOPE_MULTIMODAL_PATH)) {
			normalized = normalized.substring(0, normalized.length() - DASHSCOPE_MULTIMODAL_PATH.length());
		}
		if (normalized.endsWith(DASHSCOPE_COMPATIBLE_MODE_PATH)) {
			normalized = normalized.substring(0, normalized.length() - DASHSCOPE_COMPATIBLE_MODE_PATH.length())
					+ DASHSCOPE_NATIVE_API_PATH;
		}
		return normalized;
	}

	/**
	 * 校验 Qwen Image 3 单次输出数量。
	 *
	 * @author scott
	 * @since 2026-08-07 【无号】限制Qwen Image 3单次输出数量
	 */
	private int normalizeQwenImageCount(Integer imageCount) {
		int count = imageCount == null || imageCount < 1 ? 1 : imageCount;
		if (count > QWEN_IMAGE_3_MAX_OUTPUT_COUNT) {
			throw new IllegalArgumentException("Qwen Image 3 单次最多生成6张图片");
		}
		return count;
	}

    /**
     * 根据图片内容和提示词生成图片（目前底层仅支持千问）
     *
     * @param prompt
     * @param originalImages
     * @param params
     * @return
     */
    public List<Map<String,Object>> imageEdit(String prompt, List<String> originalImages , AIParams params) {
        if(!AiModelFactory.AIMODEL_TYPE_QWEN.equalsIgnoreCase(params.getProvider())){
            log.info("除万象模型其他模型暂不支持图生图模式，使用文生图模式，当前模型：" + params.modelName);
            return this.imageGenerate(prompt,params);
        }
        params = ensureParams(params);
        if (null == params) {
            throw new IllegalArgumentException("大语言模型参数为空");
        }
        if (CollectionUtils.isEmpty(originalImages)) {
            throw new IllegalArgumentException("原始图片不能为空");
        }

        AiModelOptions options = params.toModelOptions();
        Integer imageCount = params.imageCount;
		//update-begin---author:scott ---date:20260807  for：【无号】Qwen Image 3按供应商路由多模态接口-----------
		if (isQwenImage3Model(options.getProvider(), options.getModelName())) {
			return imageEditByQwenMultimodal(prompt, originalImages, options, imageCount);
		}
		//update-end---author:scott ---date:20260807  for：【无号】Qwen Image 3按供应商路由多模态接口-----------
        //通义万象2.1和2.5走单独的配置
        if (QwenImageModelEnum.WANX_2_1_IMAGE_EDIT.getModelName().equals(options.getModelName()) || QwenImageModelEnum.WAN_2_5_I2I_PREVIEW.getModelName().equals(options.getModelName())) {
            return imageEditQwen(prompt, originalImages, options, imageCount);
        }

        return imageEditDefault(prompt, originalImages.get(0), options, imageCount);
    }

    /**
     * 处理其他的图文编辑
     * 
     * @param prompt
     * @param originalImageBase64
     * @param options
     * @param imageCount
     * @return
     */
    private List<Map<String, Object>> imageEditDefault(String prompt, String originalImageBase64, AiModelOptions options, Integer imageCount) {
        // 处理base64前缀，Image.builder().base64Data()通常需要纯base64
        if (originalImageBase64.contains("base64,")) {
            originalImageBase64 = originalImageBase64.split("base64,")[1];
        }

        ImageModel imageModel = AiModelFactory.createImageModel(options);
        List<Map<String,Object>> result = new ArrayList<>();
        // 构造输入图片
        Image inputImage = Image.builder().base64Data(originalImageBase64).build();
        int count = (imageCount == null || imageCount < 1) ? 1 : imageCount;

        try {
            for (int i = 0; i < count; i++) {
                Response<Image> response = imageModel.edit(inputImage, prompt);
                Image image = response.content();
                Map<String,Object> item = new HashMap<>();
                if (StringUtils.isNotEmpty(image.base64Data())) {
                    item.put("type","base64");
                    String base64 = image.base64Data();
                    if (!base64.startsWith("data:")) {
                        base64 = "data:image/png;base64," + base64;
                    }
                    item.put("value", base64);
                } else if (image.url() != null) {
                    item.put("type","http");
                    item.put("value", image.url().toString());
                }
                result.add(item);
            }
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }

        return result;
    }

    /**
     * 处理千问的图片编辑
     *
     * @param prompt
     * @param originalImages
     * @param options
     * @param imageCount
     * @return
     */
    private List<Map<String, Object>> imageEditQwen(String prompt, List<String> originalImages, AiModelOptions options, Integer imageCount) {
        // 校验并调整图片尺寸
        originalImages = checkAndResizeImage(originalImages);

        int count = (imageCount == null || imageCount < 1) ? 1 : imageCount;
        List<Map<String, Object>> result = new ArrayList<>();

        try {

            ImageSynthesisParam param = ImageSynthesisParam.builder()
                    .apiKey(options.getApiKey())
                    .model(StringUtils.isNotEmpty(options.getModelName()) ? options.getModelName() : QwenImageModelEnum.WAN_2_5_I2I_PREVIEW.getModelName())
                    .prompt(prompt)
                    .function(ImageSynthesis.ImageEditFunction.DESCRIPTION_EDIT)
                    .n(count)
                    .build();
            if (StringUtils.isNotEmpty(options.getImageSize())) {
                param.setSize(options.getImageSize());
            } else {
                param.setSize("1024*1024");
            }

            if (QwenImageModelEnum.WAN_2_5_I2I_PREVIEW.getModelName().equals(options.getModelName())) {
                param.setImages(originalImages);
            }
            
            if (QwenImageModelEnum.WANX_2_1_IMAGE_EDIT.getModelName().equals(options.getModelName())) {
                param.setBaseImageUrl(originalImages.get(0));
            }

            ImageSynthesis imageSynthesis = new ImageSynthesis("text2image", options.getBaseUrl());
            ImageSynthesisResult resultResponse = imageSynthesis.call(param);

            if (resultResponse.getOutput() != null && resultResponse.getOutput().getResults() != null) {
                for (Map<String, String> item : resultResponse.getOutput().getResults()) {
                    Map<String, Object> map = new HashMap<>();
                    if (item.containsKey("url")) {
                        map.put("type", "http");
                        map.put("value", item.get("url"));
                    } else if (item.containsKey("b64_json")) {
                        map.put("type", "base64");
                        String b64 = item.get("b64_json");
                        if (!b64.startsWith("data:")) {
                            b64 = "data:image/png;base64," + b64;
                        }
                        map.put("value", b64);
                    }
                    result.add(map);
                }
            } else {
                log.error(resultResponse.getOutput().getMessage());
            }
            return result;
        } catch (Exception e) {
            throw new RuntimeException("Qwen image edit failed: " + e.getMessage(), e);
        }
    }

    /**
     * 校验并调整图片尺寸
     *
     * @param base64ImageList
     * @return
     */
    private List<String> checkAndResizeImage(List<String> base64ImageList) {
        List<String> result = new ArrayList<>();
        for (String base64Image : base64ImageList) {
            try {
                String base64Data = base64Image;
                if (base64Image.contains("base64,")) {
                    String[] parts = base64Image.split("base64,");
                    base64Data = parts[1];
                }

                // 处理可能存在的换行符
                base64Data = base64Data.replaceAll("[\\s\r\n]", "");

                byte[] imageBytes = Base64.getDecoder().decode(base64Data);
                ByteArrayInputStream bis = new ByteArrayInputStream(imageBytes);
                BufferedImage image = ImageIO.read(bis);
                if (image == null) {
                    log.warn("ImageIO read failed, use original image");
                    result.add("data:image/png;base64," + base64Image);
                    continue;
                }

                int width = image.getWidth();
                int height = image.getHeight();

                // 阿里Qwen要求高度在 512~4096 之间
                int minHeight = 512;
                int maxHeight = 4096;

                if (height >= minHeight && height <= maxHeight) {
                    result.add("data:image/png;base64," + base64Image);
                    continue;
                }

                int newHeight = height;
                int newWidth = width;

                if (height < minHeight) {
                    newHeight = minHeight;
                    newWidth = (int) (width * ((double) minHeight / height));
                } else if (height > maxHeight) {
                    newHeight = maxHeight;
                    newWidth = (int) (width * ((double) maxHeight / height));
                }

                log.info("Resize image from {}x{} to {}x{}", width, height, newWidth, newHeight);

                BufferedImage outputImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = outputImage.createGraphics();
                g2d.drawImage(image.getScaledInstance(newWidth, newHeight, 4), 0, 0, null);
                g2d.dispose();

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                // 统一转为png
                ImageIO.write(outputImage, "png", bos);
                byte[] newBytes = bos.toByteArray();

                String newBase64 = Base64.getEncoder().encodeToString(newBytes);
                result.add("data:image/png;base64," + newBase64);

            } catch (Exception e) {
                log.error("Check and resize image failed: {}", e.getMessage());
                result.add("data:image/png;base64," + base64Image);
            }
        }
        return result;
    }
}
