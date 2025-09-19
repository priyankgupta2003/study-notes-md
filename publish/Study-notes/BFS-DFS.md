#BFS  
```
def bfs(graph,node):  
    visited=[]  
    queue=[]      
    visited.append(node)  
    queue.append(node)  
      
    while queue:  
        s=queue.pop(0)  
          
        for x in graph[s]:  
            if x not in visited:  
                visited.append(x)  
                queue.append(x)  
    return visited#DFS  
```

#DFS
```
def dfs(graph,node):  
    visited=[]  
    queue=[]  
      
    queue.append(node)  
    visited.append(node)  
      
    while queue:  
        s=queue.pop()  
        print(s)        for x in graph[s][::-1]:  
            if x not in visited:  
                visited.append(x)  
                queue.append(x)
```
