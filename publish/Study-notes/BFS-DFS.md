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

#2pointer
```
Template:
def two_pointers_opposite(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]  # or process the pair
        elif current_sum < target:
            left += 1  # need larger sum
        else:
            right -= 1  # need smaller sum
    
    return []  # no solution found
```

#Fast-slow_pointer

```
def same_direction_pointers(arr):
    slow = 0  # points to position where next valid element should go
    
    for fast in range(len(arr)):
        if is_valid(arr[fast]):  # condition for keeping element
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # new length or process remaining elements
```

#3pointer-BinarySearch

```
def dutch_flag_partition(arr, pivot):
    low, mid, high = 0, 0, len(arr) - 1
    
    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:  # arr[mid] > pivot
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
            # Don't increment mid here!
```

#**Fast & Slow Pointers (Floyd's Cycle Detection)**

```
def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    
    # Phase 1: Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            # Phase 2: Find cycle start
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return fast  # or True if just detecting cycle
    
    return False
```

#finding-middle-node

```
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # middle node
```




