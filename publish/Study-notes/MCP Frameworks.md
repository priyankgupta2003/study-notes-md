### **Introduction**
MCP enables AI models to connect with external data sources, tools, and environments, allowing for the seamless transfer of information and capabilities between AI systems and the broader digital world. This interoperability is crucial for the growth and adoption of truly useful AI applications.

**Need of MCP:**
The AI / LLM models are often limited by their training data and lack of access to real-time information or specialized tools. This limitation hinders the potential of AI systems to provide truly relevant, accurate, and helpful responses in many scenarios.

**Benefits of MCP:**
- **users** enjoy simpler and more consistent experiences across AI applications
- **AI application developers** gain easy integration with a growing ecosystem of tools and data sources
- **tool and data providers** need only create a single implementation that works with multiple AI applications
- the broader ecosystem benefits from increased interoperability, innovation, and reduced fragmentation

**The Integration Problem** 
The M×N Integration Problem refers to the challenge of connecting M different AI applications to N different external tools or data sources without a standardized approach.

Without MCP
Without a protocol like MCP, developers would need to create M×N custom integrations—one for each possible pairing of an AI application with an external capability.

![[Pasted image 20250826221618.png]]

Each AI application would need to integrate with each tool/data source individually. This is very complex and expensive process which introduces a lot of friction for developers, and high maintenance costs.

Once we have multiple models and multiple tools, the number of integrations becomes too large to manage, each with its own unique interface.

![[Pasted image 20250826221824.png]]


**With MCP** (M+N Solution)

MCP transforms this into an M+N problem by providing a standard interface: each AI application implements the client side of MCP once, and each tool/data source implements the server side once. This dramatically reduces integration complexity and maintenance burden.

![[Pasted image 20250826221915.png]]

Each AI application implements the client side of MCP once, and each tool/data source implements the server side once.


### **Core MCP Terminology**

MCP is a standard like HTTP or USB-C, and is a protocol for connecting AI applications to external tools and data sources. Therefore, using standard terminology is crucial to making the MCP work effectively.

![[Pasted image 20250826222528.png]]

Components

1 Host: The user-facing AI application that end-users interact with directly. 
Examples include Anthropic’s Claude Desktop, AI-enhanced IDEs like Cursor, inference libraries like Hugging Face Python SDK, or custom applications built in libraries like LangChain or smolagents. 
**Hosts initiate connections to MCP Servers and orchestrate the overall flow between user requests, LLM processing, and external tools.**


2 Client:  A component within the host application that manages communication with a specific MCP Server. 
Each Client maintains a 1:1 connection with a single Server, handling the protocol-level details of MCP communication and acting as an intermediary between the Host’s logic and the external Server.

3 Server: An external program or service that exposes capabilities (Tools, Resources, Prompts) via the MCP protocol.

Capabilities

| Capability    | Description                                                                                                                                                                  | Example                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Tools**     | Executable functions that the AI model can invoke to perform actions or retrieve computed data. Typically relating to the use case of the application.                       | A tool for a weather application might be a function that returns the weather in a specific location. |
| **Resources** | Read-only data sources that provide context without significant computation.                                                                                                 | A researcher assistant might have a resource for scientific papers.                                   |
| **Prompts**   | Pre-defined templates or workflows that guide interactions between users, AI models, and the available capabilities.                                                         | A summarization prompt.                                                                               |
| **Sampling**  | Server-initiated requests for the Client/Host to perform LLM interactions, enabling recursive actions where the LLM can review generated content and make further decisions. | A writing application reviewing its own output and decides to refine it further.                      |

The following diagram shows the collective capabilities to a use case for a code agent.

![[Pasted image 20250826223318.png]] 


This application might use their MCP entities in the following way:

| Entity   | Name             | Description                                                                   |
| -------- | ---------------- | ----------------------------------------------------------------------------- |
| Tool     | Code Interpreter | A tool that can execute code that the LLM writes.                             |
| Resource | Documentation    | A resource that contains the documentation of the application.                |
| Prompt   | Code Style       | A prompt that guides the LLM to generate code.                                |
| Sampling | Code Review      | A sampling that allows the LLM to review the code and make further decisions. |


### **Architecture**
**Host, Client, and Server**

The MCP is built on client-server architecture that enables structured communication between AI models and external systems.

![[Pasted image 20250826223602.png]] 


The MCP architecture consists of three primary components, each with well-defined roles and responsibilities: Host, Client, and Server

**Host**

The **Host** is the user-facing AI application that end-users interact with directly.

Examples include:

- AI Chat apps like OpenAI ChatGPT or Anthropic’s Claude Desktop
- AI-enhanced IDEs like Cursor, or integrations to tools like Continue.dev
- Custom AI agents and applications built in libraries like LangChain or smolagents

The Host’s responsibilities include:

- Managing user interactions and permissions
- Initiating connections to MCP Servers via MCP Clients
- Orchestrating the overall flow between user requests, LLM processing, and external tools
- Rendering results back to users in a coherent format

In most cases, users will select their host application based on their needs and preferences. For example, a developer may choose Cursor for its powerful code editing capabilities, while domain experts may use custom applications built in smolagents.


**Client**

The **Client** is a component within the Host application that manages communication with a specific MCP Server. Key characteristics include:

- Each Client maintains a 1:1 connection with a single Server
- Handles the protocol-level details of MCP communication
- Acts as the intermediary between the Host’s logic and the external Server

 **Server**

The **Server** is an external program or service that exposes capabilities to AI models via the MCP protocol. Servers:

- Provide access to specific external tools, data sources, or services
- Act as lightweight wrappers around existing functionality
- Can run locally (on the same machine as the Host) or remotely (over a network)
- Expose their capabilities in a standardized format that Clients can discover and use

**Communication Flow**
How the components interact in a MCP workflow:
1. **User Interaction**: The user interacts with the **Host** application, expressing an intent or query.
2. **Host Processing**: The **Host** processes the user’s input, potentially using an LLM to understand the request and determine which external capabilities might be needed.
3. **Client Connection**: The **Host** directs its **Client** component to connect to the appropriate Server(s).
4. **Capability Discovery**: The **Client** queries the **Server** to discover what capabilities (Tools, Resources, Prompts) it offers.
5. **Capability Invocation**: Based on the user’s needs or the LLM’s determination, the Host instructs the **Client** to invoke specific capabilities from the **Server**.
6. **Server Execution**: The **Server** executes the requested functionality and returns results to the **Client**.
7. **Result Integration**: The **Client** relays these results back to the **Host**, which incorporates them into the context for the LLM or presents them directly to the user.

Advantages of this architecture:
	1. modular architecture
	2. A single host can connect to multiple servers simultaneously via different clients,
	3. new servers can be added to the ecosystem without changing the existing hosts.
	4. easily composed across different servers.

The architecture might appear simple, but its power lies in the standardization of the communication protocol and the clear separation of responsibilities between components. This design allows for a cohesive ecosystem where AI models can seamlessly connect with an ever-growing array of external tools and data sources.

Notes:
These interaction patterns are guided by several key principles that shape the design and evolution of MCP. The protocol emphasizes **standardization** by providing a universal protocol for AI connectivity, while maintaining **simplicity** by keeping the core protocol straightforward yet enabling advanced features. **Safety** is prioritized by requiring explicit user approval for sensitive operations, and discoverability enables dynamic discovery of capabilities. The protocol is built with **extensibility** in mind, supporting evolution through versioning and capability negotiation, and ensures **interoperability** across different implementations and environments.



### **Communication Protocol**

**JSON-RPC: The Foundation**
MCP uses **JSON-RPC 2.0**, at its core, as the message format for all communication between clients and servers. 
JSON-RPC is a lightweight remote procedure call protocol encoded in JSON, which makes it:
- Human-readable and easy to debug
- Language-agnostic, supporting implementation in any programming environment
- Well-established, with clear specifications and widespread adoption

![[Pasted image 20250827110009.png]] 

The protocol defines 3 types of messages:

1. Requests
		sent from Client to Server to initiate an operation. A request message includes:
			- A unique id
			- The method name to invoke (tools/ functions)
			- Parameters for the method (if any)
		eg: 
		{
			  "jsonrpc": "2.0",
			  "id": 1,
			  "method": "tools/call",
			  "params": {
			    "name": "weather",
			    "arguments": {
			      "location": "San Francisco"
			    }
			  }
			}


2. Responses
		Sent from server to client in reply to the request. A response message includes
				- same id as corresponding request
				- either a result (for success) or an error (for failure)
			eg:
			success reponse
			{
			  "jsonrpc": "2.0",
			  "id": 1,
			  "result": {
				"temperature": 62,
				"conditions": "Partly cloudy"
			  }
			}
			Error Response:
			{
			  "jsonrpc": "2.0",
			  "id": 1,
			  "result": {
			    "temperature": 62,
			    "conditions": "Partly cloudy"
			  }
			}
3. Notifications
		One - way messages that don't require a response. Typically sent from Server to Client to provide updates or notifications about events.
		eg:
		```{
		  "jsonrpc": "2.0",
		  "method": "progress",
		  "params": {
			"message": "Processing data...",
			"percent": 50
		  }
		}```


**Transport Mechanisms**
JSON - RPC defines the message format, but MCP also specifies how these messages are transported between clients and servers.
	1. Stdio (Standard Input/Ouput) : used for local communication, where the client and server run on the same machine.
		(The Host application launches the Server as a subprocess and communicates with it by writing to its standard input (stdin) and reading from its standard output (stdout).)
		**usecase**: used for local tools like file system access or running local scripts.
		**Benefit** : The main **advantage** of this transport are that it's simple, no network configuration required, and securely sandboxed by the operating system.
	2.  HTTP + SSE (Server - Sent Events) / Streamable HTTP: 
	The HTTP+SSE transport is used for remote communication, where the Client and Server might be on different machines:
	Communication happens over HTTP, with the Server using Server-Sent Events (SSE) to push updates to the Client over a persistent connection.
	The main **Advantages** of this transport are that it works across networks, enables integration with web services, and is compatible with serverless environments.

**The Interaction Lifecycle:**

1. Initialization :  Client connects to the Server and they exchange protocol versions and capabilities, and the Server responds with its supported protocol version and capabilities. (client confirmation - notification message)
2. Discovery : The Client requests information about available capabilities and the Server responds with a list of available tools. (repeated for each tool/resource/prompt)
3. Execution : The Client invokes capabilities based on the Host’s needs.
4. Termination : The connection is gracefully closed when no longer needed and the Server acknowledges the shutdown request. (exit message)


**Protocol Evolution**
- designed to be extensible and adaptable
-  The initialization phase includes version negotiation, allowing for backward compatibility as the protocol evolves
- capability discovery enables Clients to adapt to the specific features each Server offers

### **Understanding MCP Capabilities**

1. Tools : executable functions or actions that the AI model can invoke
	- Control: tools are **model-controlled** i.e. AI model can decide when to call them based on the user's request and context
	- Safety: Due to their ability to perform actions with side effects, tool execution can be dangerous. Therefore, they typically require explicit user approval.
	- Use Cases: Sending messages, creating tickets, querying APIs performing calculations.
	
2. Resources: Resources provide read-only access to data sources, allowing the AI model to retrieve context without executing complex logic
	- **Control**: Resources are **application-controlled**, meaning the Host application typically decides when to access them.
	- **Nature**: They are designed for data retrieval with minimal computation, similar to GET endpoints in REST APIs.
	- **Safety**: Since they are read-only, they typically present lower security risks than Tools.
	- **Use Cases**: Accessing file contents, retrieving database records, reading configuration information.
	
3.  Prompts: Prompts are predefined templates or workflows that guide the interaction between the user, the AI model, and the Server’s capabilities.
	- **Control**: Prompts are **user-controlled**, often presented as options in the Host application’s UI.
	- **Purpose**: They structure interactions for optimal use of available Tools and Resources.
	- **Selection**: Users typically select a prompt before the AI model begins processing, setting context for the interaction.
	- **Use Cases**: Common workflows, specialized task templates, guided interactions.

4. Sampling: Sampling allows Servers to request the Client (specifically, the Host application) to perform LLM interactions.

	- **Control**: Sampling is **server-initiated** but requires Client/Host facilitation.
	- **Purpose**: It enables server-driven agentic behaviors and potentially recursive or multi-step interactions.
	- **Safety**: Like Tools, sampling operations typically require user approval.
	- **Use Cases**: Complex multi-step tasks, autonomous agent workflows, interactive processes.
	
The sampling flow follows these steps:

1. Server sends a `sampling/createMessage` request to the client
2. Client reviews the request and can modify it
3. Client samples from an LLM
4. Client reviews the completion
5. Client returns the result to the server

**How Capabilities Work Together**

|Capability|Controlled By|Direction|Side Effects|Approval Needed|Typical Use Cases|
|---|---|---|---|---|---|
|Tools|Model (LLM)|Client → Server|Yes (potentially)|Yes|Actions, API calls, data manipulation|
|Resources|Application|Client → Server|No (read-only)|Typically no|Data retrieval, context gathering|
|Prompts|User|Server → Client|No|No (selected by user)|Guided workflows, specialized templates|
|Sampling|Server|Server → Client → Server|Indirectly|Yes|Multi-step tasks, agentic behaviors|

The distinction between these primitives provides a clear structure for MCP interactions, enabling AI models to access information, perform actions, and engage in complex workflows while maintaining appropriate control boundaries.


 **Discovery Process**

One of MCP’s key features is dynamic capability discovery. When a Client connects to a Server, it can query the available Tools, Resources, and Prompts through specific list methods:

- `tools/list`: Discover available Tools
- `resources/list`: Discover available Resources
- `prompts/list`: Discover available Prompts

This dynamic discovery mechanism allows Clients to adapt to the specific capabilities each Server offers without requiring hardcoded knowledge of the Server’s functionality.

## **MCP SDK**


The Model Context Protocol provides official SDKs for both JavaScript, Python and other languages. This makes it easy to implement MCP clients and servers in your applications. These SDKs handle the low-level protocol details, allowing you to focus on building your application's capabilities.

 SDK Overview

Both SDKs provide similar core functionality, following the MCP protocol specification we discussed earlier. They handle:

- Protocol-level communication
- Capability registration and discovery
- Message serialization/deserialization
- Connection management
- Error handling

 Core Primitives Implementation

Let's explore how to implement each of the core primitives (Tools, Resources, and Prompts) using both SDKs.


```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Weather Service")

# Tool implementation
@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a specified location."""
    return f"Weather in {location}: Sunny, 72°F"

# Resource implementation
@mcp.resource("weather://{location}")
def weather_resource(location: str) -> str:
    """Provide weather data as a resource."""
    return f"Weather data for {location}: Sunny, 72°F"

# Prompt implementation
@mcp.prompt()
def weather_report(location: str) -> str:
    """Create a weather report prompt."""
    return f"""You are a weather reporter. Weather report for {location}?"""


# Run the server
if __name__ == "__main__":
    mcp.run()
```

Once you have your server implemented, you can start it by running the server script.

```bash
mcp dev server.py
```


This will initialize a development server running the file `server.py`. And log the following output:

```bash
Starting MCP inspector...
⚙️ Proxy server listening on port 6277
Spawned stdio transport
Connected MCP client to backing server transport
Created web app transport
Set up MCP proxy
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
```

You can then open the MCP Inspector at [http://127.0.0.1:6274](http://127.0.0.1:6274) to see the server's capabilities and interact with them.

You'll see the server's capabilities and the ability to call them via the UI.

![MCP Inspector](https://huggingface.co/datasets/mcp-course/images/resolve/main/unit1/6.png)

 MCP SDKs

MCP is designed to be language-agnostic, and there are official SDKs available for several popular programming languages:

| Language   | Repository                                                                                               | Maintainer(s)       | Status           |
| ---------- | -------------------------------------------------------------------------------------------------------- | ------------------- | ---------------- |
| TypeScript | [github.com/modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk) | Anthropic           | Active           |
| Python     | [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)         | Anthropic           | Active           |
| Java       | [github.com/modelcontextprotocol/java-sdk](https://github.com/modelcontextprotocol/java-sdk)             | Spring AI (VMware)  | Active           |
| Kotlin     | [github.com/modelcontextprotocol/kotlin-sdk](https://github.com/modelcontextprotocol/kotlin-sdk)         | JetBrains           | Active           |
| C#         | [github.com/modelcontextprotocol/csharp-sdk](https://github.com/modelcontextprotocol/csharp-sdk)         | Microsoft           | Active (Preview) |
| Swift      | [github.com/modelcontextprotocol/swift-sdk](https://github.com/modelcontextprotocol/swift-sdk)           | loopwork-ai         | Active           |
| Rust       | [github.com/modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)             | Anthropic/Community | Active           |
| Dart       | [https://github.com/leehack/mcp_dart](https://github.com/leehack/mcp_dart)                               | Flutter Community   | Active           |

These SDKs provide language-specific abstractions that simplify working with the MCP protocol, allowing you to focus on implementing the core logic of your servers or clients rather than dealing with low-level protocol details.


## **MCP Clients**

Now that we have a basic understanding of the Model Context Protocol, we can explore the essential role of MCP Clients in the Model Context Protocol ecosystem.

 In this part of Unit 1, we'll explore the essential role of MCP Clients in the Model Context Protocol ecosystem.

In this section, you will:

* Understand what MCP Clients are and their role in the MCP architecture
* Learn about the key responsibilities of MCP Clients
* Explore the major MCP Client implementations
* Discover how to use Hugging Face's MCP Client implementation
* See practical examples of MCP Client usage

<Tip>

In this page we're going to show examples of how to set up MCP Clients in a few different ways using the JSON notation. For now, we will use *examples* like `path/to/server.py` to represent the path to the MCP Server. In the next unit, we'll implement this with real MCP Servers.  

For now, focus on understanding the MCP Client notation. We'll implement the MCP Servers in the next unit.

</Tip>

Understanding MCP Clients

MCP Clients are crucial components that act as the bridge between AI applications (Hosts) and external capabilities provided by MCP Servers. Think of the Host as your main application (like an AI assistant or IDE) and the Client as a specialized module within that Host responsible for handling MCP communications.

User Interface Client

Let's start by exploring the user interface clients that are available for the MCP.

### Chat Interface Clients

Anthropic's Claude Desktop stands as one of the most prominent MCP Clients, providing integration with various MCP Servers.

### Interactive Development Clients

Cursor's MCP Client implementation enables AI-powered coding assistance through direct integration with code editing capabilities. It supports multiple MCP Server connections and provides real-time tool invocation during coding, making it a powerful tool for developers.

Continue.dev is another example of an interactive development client that supports MCP and connects to an MCP server from VS Code.

## Configuring MCP Clients

Now that we've covered the core of the MCP protocol, let's look at how to configure your MCP servers and clients.

Effective deployment of MCP servers and clients requires proper configuration. 

<Tip>

The MCP specification is still evolving, so the configuration methods are subject to evolution. We'll focus on the current best practices for configuration.

</Tip>

### MCP Configuration Files

MCP hosts use configuration files to manage server connections. These files define which servers are available and how to connect to them.

Fortunately, the configuration files are very simple, easy to understand, and consistent across major MCP hosts.

#### `mcp.json` Structure

The standard configuration file for MCP is named `mcp.json`. Here's the basic structure:

This is the basic structure of the `mcp.json` can be passed to applications like Claude Desktop, Cursor, or VS Code.

```json
{
  "servers": [
    {
      "name": "Server Name",
      "transport": {
        "type": "stdio|sse",
        // Transport-specific configuration
      }
    }
  ]
}
```

In this example, we have a single server with a name and a transport type. The transport type is either `stdio` or `sse`.

#### Configuration for stdio Transport

For local servers using stdio transport, the configuration includes the command and arguments to launch the server process:

```json
{
  "servers": [
    {
      "name": "File Explorer",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/file_explorer_server.py"] // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}
```

Here, we have a server called "File Explorer" that is a local script.

#### Configuration for HTTP+SSE Transport

For remote servers using HTTP+SSE transport, the configuration includes the server URL:

```json
{
  "servers": [
    {
      "name": "Remote API Server",
      "transport": {
        "type": "sse",
        "url": "https://example.com/mcp-server"
      }
    }
  ]
}
```

#### Environment Variables in Configuration

Environment variables can be passed to server processes using the `env` field. Here's how to access them in your server code:

<hfoptions id="env-variables">
<hfoption id="python">

In Python, we use the `os` module to access environment variables:

```python
import os

# Access environment variables
github_token = os.environ.get("GITHUB_TOKEN")
if not github_token:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# Use the token in your server code
def make_github_request():
    headers = {"Authorization": f"Bearer {github_token}"}
    # ... rest of your code
```


The corresponding configuration in `mcp.json` would look like this:

```json
{
  "servers": [
    {
      "name": "GitHub API",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/github_server.py"], // This is an example, we'll use a real server in the next unit
        "env": {
          "GITHUB_TOKEN": "your_github_token"
        }
      }
    }
  ]
}
```

### Configuration Examples

Let's look at some real-world configuration scenarios:

#### Scenario 1: Local Server Configuration

In this scenario, we have a local server that is a Python script which could be a file explorer or a code editor.

```json
{
  "servers": [
    {
      "name": "File Explorer",
      "transport": {
        "type": "stdio",
        "command": "python",
        "args": ["/path/to/file_explorer_server.py"] // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}
```

#### Scenario 2: Remote Server Configuration

In this scenario, we have a remote server that is a weather API.

```json
{
  "servers": [
    {
      "name": "Weather API",
      "transport": {
        "type": "sse",
        "url": "https://example.com/mcp-server" // This is an example, we'll use a real server in the next unit
      }
    }
  ]
}
```

Proper configuration is essential for successfully deploying MCP integrations. By understanding these aspects, you can create robust and reliable connections between AI applications and external capabilities.

In the next section, we'll explore the ecosystem of MCP servers available on Hugging Face Hub and how to publish your own servers there. 

## Tiny Agents Clients

Now, let's explore how to use MCP Clients within code.

You can also use tiny agents as MCP Clients to connect directly to MCP servers from your code. Tiny agents provide a simple way to create AI agents that can use tools from MCP servers.

Tiny Agent can run MCP servers with a command line environment. To do this, we will need to install `npm` and run the server with `npx`. **We'll need these for both Python and JavaScript.**

Let's install `npx` with `npm`. If you don't have `npm` installed, check out the [npm documentation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

### Setup

First, we will need to install `npx` if you don't have it installed. You can do this with the following command:

```bash
# install npx
npm install -g npx
```

Then, we will need to install the huggingface_hub package with the MCP support. This will allow us to run MCP servers and clients.

```bash
pip install "huggingface_hub[mcp]>=0.32.0"
```

Then, we will need to log in to the Hugging Face Hub to access the MCP servers. You can do this with the `huggingface-cli` command line tool. You will need a [login token](https://huggingface.co/docs/huggingface_hub/v0.32.3/en/quick-start#authentication) to do this.

```bash
huggingface-cli login
```

<hfoptions id="language">
<hfoption id="python">

### Connecting to MCP Servers

Now, let's create an agent configuration file `agent.json`.

```json
{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"]
            }
        }
    ]
}
``` 

In this configuration, we are using the `@playwright/mcp` MCP server. This is an MCP server that can control a browser with Playwright.

Now you can run the agent:

```bash
tiny-agents run agent.json
```
</hfoption>
<hfoption id="javascript">

First, install the tiny agents package with [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

```bash
npm install @huggingface/tiny-agents
```

### Connecting to MCP Servers

Make an agent project directory and create an `agent.json` file.

```bash
mkdir my-agent
touch my-agent/agent.json
```

Create an agent configuration file at `my-agent/agent.json`:

```json
{
	"model": "Qwen/Qwen2.5-72B-Instruct",
	"provider": "nebius",
	"servers": [
		{
			"type": "stdio",
			"config": {
				"command": "npx",
				"args": ["@playwright/mcp@latest"]
			}
		}
	]
}
```

Now you can run the agent:

```bash
npx @huggingface/tiny-agents run ./my-agent
```

</hfoption>
</hfoptions>

In the video below, we run the agent and ask it to open a new tab in the browser.

The following example shows a web-browsing agent configured to use the [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model via Nebius inference provider, and it comes equipped with a playwright MCP server, which lets it use a web browser! The agent config is loaded specifying [its path in the `tiny-agents/tiny-agents`](https://huggingface.co/datasets/tiny-agents/tiny-agents/tree/main/celinah/web-browser) Hugging Face dataset.

<video controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/python-tiny-agents/web_browser_agent.mp4" type="video/mp4">
</video>

When you run the agent, you'll see it load, listing the tools it has discovered from its connected MCP servers. Then, it's ready for your prompts!

Prompt used in this demo:

> do a Web Search for HF inference providers on Brave Search and open the first result and then give me the list of the inference providers supported on Hugging Face 


