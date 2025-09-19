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

#LCS

```
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
       
        dp = [[0] * (len(text2)+1) for _ in range(len(text1)+1)]
        
        for i in range(1,len(text1)+1):
            for j in range(1,len(text2)+1):
                
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[len(text1)][len(text2)]
```

