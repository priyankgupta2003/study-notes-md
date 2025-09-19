
## **1. Two Pointers Patterns**

### **1.1 Opposite Direction Two Pointers**

**Core Concept**: Use two pointers starting from opposite ends of an array, moving them towards each other based on certain conditions.

**Key Insights**:

- Usually works on **sorted arrays** or when you need to find pairs/triplets
    
- **Time Complexity**: O(n) - much better than nested loops O(n²)
    
- **Space Complexity**: O(1) - no extra space needed
    
- **Movement Strategy**: Move left pointer right or right pointer left based on current sum vs target
    

**When to Use**:

- Finding pairs with a target sum in sorted array
    
- Checking if array/string is a palindrome
    
- Container/water problems where you need maximum area
    
- Three sum problems (fix one element, use two pointers for remaining)
    

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

**Example Problem Analysis** - Two Sum II:

- Array is sorted, so we can use two pointers
    
- If sum is too small, move left pointer (increase sum)
    
- If sum is too large, move right pointer (decrease sum)
    
- This avoids O(n²) brute force approach
    

### **1.2 Same Direction Two Pointers (Fast-Slow)**

**Core Concept**: Both pointers move in the same direction, but at different speeds or with different conditions.

**Key Insights**:

- **Fast pointer** explores ahead or moves unconditionally
    
- **Slow pointer** moves only when certain conditions are met
    
- Used for **in-place array modifications** without extra space
    
- **Partitioning** elements based on conditions
    

**When to Use**:

- Removing duplicates from sorted array
    
- Moving zeros to end
    
- Partitioning arrays (Dutch flag problem)
    
- Any problem requiring in-place array rearrangement
    

**Template**:

```
def same_direction_pointers(arr):
    slow = 0  # points to position where next valid element should go
    
    for fast in range(len(arr)):
        if is_valid(arr[fast]):  # condition for keeping element
            arr[slow] = arr[fast]
            slow += 1
    
    return slow  # new length or process remaining elements
```

**Example Problem Analysis** - Remove Duplicates:

- Fast pointer scans all elements
    
- Slow pointer tracks position for next unique element
    
- Only advance slow when we find a new unique element
    
- Overwrite duplicates in-place
    

### **1.3 Three Pointers**

**Core Concept**: Use three pointers to handle problems requiring three-way partitioning or finding triplets.

**Key Insights**:

- **Dutch Flag Algorithm**: Partition array into three sections
    
- **3Sum Pattern**: Fix one element, use two pointers for remaining two
    
- **Three-way partitioning**: Less than, equal to, greater than a pivot
    

**When to Use**:

- Sorting arrays with three distinct values (0s, 1s, 2s)
    
- Finding triplets with specific sum
    
- Problems requiring three-way classification
    

**Template for Dutch Flag**:

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

**Example Problem Analysis** - Sort Colors:

- Three regions: [0...low-1] for 0s, [low...mid-1] for 1s, [high+1...n-1] for 2s
    
- Mid pointer explores unknown region
    
- Swap elements to appropriate regions based on value
    

## **2. Fast & Slow Pointers (Floyd's Cycle Detection)**

### **2.1 Cycle Detection in Linked Lists**

**Core Concept**: Use two pointers moving at different speeds to detect cycles in linked structures.

**Key Insights**:

- **Fast pointer** moves 2 steps, **slow pointer** moves 1 step
    
- If there's a cycle, they will eventually meet inside the cycle
    
- **Mathematical proof**: If cycle exists, fast pointer will "lap" slow pointer
    
- **Finding cycle start**: Reset one pointer to head, move both at same speed
    

**When to Use**:

- Detecting cycles in linked lists
    
- Finding duplicate numbers in arrays (treat as implicit linked list)
    
- Happy number problems
    
- Any problem involving repeated states
    

**Template**:

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

**Example Problem Analysis** - Linked List Cycle:

- If no cycle: fast reaches end before meeting slow
    
- If cycle exists: fast will eventually catch up to slow inside cycle
    
- Distance traveled when they meet gives us cycle information
    

### **2.2 Finding Middle Element**

**Core Concept**: When fast pointer reaches end, slow pointer is at middle.

**Key Insights**:

- **Odd length**: Slow points to exact middle
    
- **Even length**: Slow points to second middle element
    
- **Modifications**: Adjust for different middle definitions
    
- **Applications**: Palindrome checking, list splitting
    

**Template**:

```
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # middle node
```

**Example Problem Analysis** - Middle of Linked List:

- Fast moves twice as fast as slow
    
- When fast reaches end, slow is at middle
    
- Can be modified for "remove nth from end" by starting fast n steps ahead
    

## **3. Sliding Window Patterns**

### **3.1 Fixed Size Sliding Window**

**Core Concept**: Maintain a window of exactly k elements, sliding it across the array.

**Key Insights**:

- **Window size** remains constant
    
- **Add new element** on right, **remove old element** on left
    
- **Precompute** first window, then slide one position at a time
    
- **Time Complexity**: O(n) - each element added and removed exactly once
    

**When to Use**:

- Finding maximum/minimum in all subarrays of size k
    
- Average of all subarrays of size k
    
- Anagram problems with fixed length
    
- Any problem with "all subarrays of size k"
    

**Template**:

```
def fixed_sliding_window(arr, k):
    if len(arr) < k:
        return []
    
    # Initialize first window
    window_sum = sum(arr[:k])
    result = [window_sum]
    
    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        result.append(window_sum)  # or process window
    
    return result
```

**Example Problem Analysis** - Maximum Average Subarray:

- Calculate sum of first k elements
    
- Slide window: subtract leftmost, add rightmost
    
- Track maximum sum seen so far
    
- Convert to average at the end
    

### **3.2 Variable Size Sliding Window - Maximum**

**Core Concept**: Expand window until condition is violated, then contract from left.

**Key Insights**:

- **Right pointer** expands window (adds elements)
    
- **Left pointer** contracts window (removes elements)
    
- **Goal**: Find the longest window satisfying condition
    
- **Two-phase**: Expand then contract in each iteration
    

**When to Use**:

- Longest substring without repeating characters
    
- Longest substring with at most k distinct characters
    
- Maximum consecutive 1s with at most k flips
    
- Any "longest/maximum" subarray problems
    

**Template**:

```
def max_sliding_window(arr, condition_checker):
    left = 0
    max_length = 0
    window_data = {}  # or whatever data structure needed
    
    for right in range(len(arr)):
        # Expand window by including arr[right]
        update_window_data(window_data, arr[right])
        
        # Contract window while condition is violated
        while not condition_checker(window_data):
            remove_from_window_data(window_data, arr[left])
            left += 1
        
        # Update result with current window size
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

**Example Problem Analysis** - Longest Substring Without Repeating Characters:

- Use hash map to track character frequencies
    
- Expand window until we see a duplicate
    
- Contract from left until no duplicates
    
- Track maximum window size seen
    

### **3.3 Variable Size Sliding Window - Minimum**

**Core Concept**: Find the shortest window that satisfies the condition.

**Key Insights**:

- **Expand** until condition is satisfied
    
- **Contract** while condition remains satisfied
    
- **Goal**: Find minimum window length
    
- **Template** slightly different from maximum version
    

**When to Use**:

- Minimum window substring covering all characters
    
- Shortest subarray with sum ≥ target
    
- Minimum window containing all elements of another array
    

**Template**:

```
def min_sliding_window(arr, target):
    left = 0
    min_length = float('inf')
    window_sum = 0
    
    for right in range(len(arr)):
        # Expand window
        window_sum += arr[right]
        
        # Contract window while condition is satisfied
        while condition_satisfied(window_sum, target):
            min_length = min(min_length, right - left + 1)
            window_sum -= arr[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

**Example Problem Analysis** - Minimum Size Subarray Sum:

- Expand window until sum ≥ target
    
- Contract from left while maintaining sum ≥ target
    
- Track minimum window size that satisfies condition
    

## **4. Merge Intervals**

### **4.1 Overlapping Intervals**

**Core Concept**: Sort intervals by start time, then merge overlapping ones.

**Key Insights**:

- **Sort first**: Always sort intervals by start time
    
- **Overlapping condition**: current.start ≤ previous.end
    
- **Merging strategy**: Extend the end time of previous interval
    
- **Non-overlapping**: Add current interval to result
    

**When to Use**:

- Merging overlapping time intervals
    
- Finding conflicts in scheduling
    
- Calculating total covered time
    
- Removing redundant intervals
    

**Template**:

```
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:  # Overlapping
            # Merge by extending end time
            last[1] = max(last[1], current[1])
        else:  # Non-overlapping
            merged.append(current)
    
    return merged
```

**Example Problem Analysis** - Merge Intervals:

- Sort by start time ensures we process in chronological order
    
- If current starts before previous ends, they overlap
    
- Extend previous interval's end to cover both intervals
    
- Continue until all intervals processed
    

### **4.2 Meeting Rooms Pattern**

**Core Concept**: Track concurrent intervals using events or priority queues.

**Key Insights**:

- **Event-based approach**: Create start/end events, sort by time
    
- **Priority queue approach**: Use min-heap to track ending times
    
- **Room allocation**: Number of concurrent meetings = rooms needed
    
- **Greedy assignment**: Always use earliest available room
    

**When to Use**:

- Meeting room scheduling problems
    
- Resource allocation problems
    
- Finding maximum concurrent intervals
    
- Calculating required capacity
    

**Template (Priority Queue)**:

```
import heapq

def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times
    min_heap = []
    
    for start, end in intervals:
        # Remove all meetings that have ended
        while min_heap and min_heap[0] <= start:
            heapq.heappop(min_heap)
        
        # Add current meeting's end time
        heapq.heappush(min_heap, end)
    
    return len(min_heap)  # Number of concurrent meetings
```

**Example Problem Analysis** - Meeting Rooms II:

- Sort meetings by start time
    
- Use min-heap to track when current meetings end
    
- For each new meeting, remove ended meetings first
    
- Heap size = concurrent meetings = rooms needed
    

## **5. Cyclic Sort**

### **5.1 Missing Number Pattern**

**Core Concept**: For arrays containing numbers in range [1,n], place each number at its correct index.

**Key Insights**:

- **Correct position**: Number n should be at index n-1 (for 1-based) or n (for 0-based)
    
- **Swapping strategy**: Keep swapping until current element is in correct position
    
- **Finding missing**: After sorting, missing number is at incorrect position
    
- **Time Complexity**: O(n) - each element moved at most once
    

**When to Use**:

- Finding missing numbers in range [1,n] or [0,n]
    
- Finding duplicates in constrained ranges
    
- Problems with arrays containing numbers in specific ranges
    
- When you need to sort but can't use comparison-based sorting
    

**Template**:

```
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        # Calculate correct position for nums[i]
        correct_pos = nums[i] - 1  # for 1-based numbering
        
        # If current element is not in correct position
        if nums[i] != nums[correct_pos]:
            # Swap to correct position
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    
    # Find missing number
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    
    return len(nums) + 1  # if no missing number in range
```

**Example Problem Analysis** - Find Missing Number:

- Each number should be at index = number - 1
    
- After cyclic sort, missing number's position will have wrong value
    
- Scan sorted array to find the mismatch
    
- Handle edge case where missing number is largest
    

## **6. In-place Reversal of LinkedList**

### **6.1 Basic Reversal**

**Core Concept**: Reverse pointers iteratively or recursively while maintaining three pointers.

**Key Insights**:

- **Three pointers**: previous, current, next
    
- **Pointer reversal**: current.next = previous
    
- **Advancing**: Move all three pointers forward
    
- **Iterative vs Recursive**: Both approaches possible
    

**When to Use**:

- Reversing entire linked list
    
- Reversing portions of linked list
    
- Swapping adjacent nodes
    
- Any problem requiring pointer direction changes
    

**Iterative Template**:

```
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        # Store next node
        next_temp = current.next
        
        # Reverse the link
        current.next = prev
        
        # Move pointers forward
        prev = current
        current = next_temp
    
    return prev  # new head

Recursive Template:
def reverse_recursive(head):
    # Base case
    if not head or not head.next:
        return head
    
    # Recursively reverse rest of list
    new_head = reverse_recursive(head.next)
    
    # Reverse current connection
    head.next.next = head
    head.next = None
    
    return new_head
```

**Example Problem Analysis** - Reverse Linked List:

- Break the problem into reversing individual links
    
- Maintain three pointers to avoid losing references
    
- Update pointers in correct order to avoid infinite loops
    
- Return new head (which was originally the tail)
    

## **7. Stack Patterns**

### **7.1 Basic Stack Operations**

**Core Concept**: Use LIFO (Last In, First Out) property for matching, validation, and parsing problems.

**Key Insights**:

- **Matching problems**: Use stack to match opening/closing pairs
    
- **Parsing**: Stack naturally handles nested structures
    
- **Validation**: Check if all opening symbols have closing pairs
    
- **State tracking**: Stack maintains current state during traversal
    

**When to Use**:

- Parentheses/bracket validation
    
- Expression parsing and evaluation
    
- Undo operations
    
- Function call management
    
- Any problem with nested structures
    

**Template**:

```
def validate_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0  # All brackets matched
```

**Example Problem Analysis** - Valid Parentheses:

- Push opening brackets onto stack
    
- For closing brackets, check if top of stack matches
    
- If all brackets matched, stack should be empty
    
- Handle edge cases: empty string, unmatched brackets
    

### **7.2 Expression Evaluation**

**Core Concept**: Use stack to evaluate mathematical expressions, handling operator precedence.

**Key Insights**:

- **Postfix notation**: Easier to evaluate using stack
    
- **Infix to postfix**: Use operator stack and output queue
    
- **Precedence handling**: Pop higher/equal precedence operators first
    
- **Parentheses**: Override normal precedence rules
    

**When to Use**:

- Calculator implementations
    
- Expression parsing
    
- Operator precedence problems
    
- Converting between notation systems
    

**Template (Postfix Evaluation)**:

```
def evaluate_postfix(tokens):
    stack = []
    
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            # Pop two operands (note the order!)
            second = stack.pop()
            first = stack.pop()
            
            # Perform operation
            if token == '+':
                result = first + second
            elif token == '-':
                result = first - second
            elif token == '*':
                result = first * second
            elif token == '/':
                result = int(first / second)  # Handle division
            
            stack.append(result)
        else:
            # Operand
            stack.append(int(token))
    
    return stack[0]
```

**Example Problem Analysis** - Basic Calculator:

- Use stack to handle operands and intermediate results
    
- Process operators when encountered or when precedence changes
    
- Handle parentheses by recursion or separate operator stack
    
- Consider edge cases: negative numbers, spaces, division by zero
    

## **8. Monotonic Stack**

### **8.1 Next Greater/Smaller Element**

**Core Concept**: Maintain a stack where elements are in monotonic (increasing/decreasing) order.

**Key Insights**:

- **Monotonic property**: Stack elements maintain specific order
    
- **When to pop**: Pop when current element breaks monotonic property
    
- **Next greater**: Use decreasing stack (pop smaller elements)
    
- **Next smaller**: Use increasing stack (pop larger elements)
    
- **Time Complexity**: O(n) - each element pushed and popped at most once
    

**When to Use**:

- Finding next greater/smaller element for each array element
    
- Daily temperatures (next warmer day)
    
- Stock span problems
    
- Histogram-related problems
    

**Template (Next Greater Element)**:

```
def next_greater_elements(nums):
    result = [-1] * len(nums)
    stack = []  # Stores indices
    
    for i in range(len(nums)):
        # Pop all smaller elements
        while stack and nums[stack[-1]] < nums[i]:
            index = stack.pop()
            result[index] = nums[i]
        
        stack.append(i)
    
    return result
```

**Template (Circular Array)**:

```
def next_greater_circular(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice for circular property
    for i in range(2 * n):
        # Pop smaller elements
        while stack and nums[stack[-1]] < nums[i % n]:
            index = stack.pop()
            result[index] = nums[i % n]
        
        # Only push indices from first iteration
        if i < n:
            stack.append(i)
    
    return result
```

**Example Problem Analysis** - Daily Temperatures:

- Use decreasing monotonic stack (indices of temperatures)
    
- For each day, pop all previous days with lower temperature
    
- The popped days have current day as their "next warmer day"
    
- Stack maintains potential candidates for future days
    

### **8.2 Histogram Pattern**

**Core Concept**: Use monotonic stack to find rectangles with maximum area.

**Key Insights**:

- **Increasing stack**: Maintain indices of increasing heights
    
- **When to calculate**: When current height is smaller than stack top
    
- **Area calculation**: height × width (width = current_index - stack_top - 1)
    
- **Boundary handling**: Add sentinel values (0) at both ends
    

**When to Use**:

- Largest rectangle in histogram
    
- Maximal rectangle in binary matrix
    
- Trapping rainwater problems
    
- Any problem involving rectangular areas
    

**Template**:

```
def largest_rectangle_histogram(heights):
    stack = []
    max_area = 0
    heights.append(0)  # Sentinel to pop all remaining elements
    
    for i, height in enumerate(heights):
        # Pop while current height is smaller
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        stack.append(i)
    
    return max_area
```

**Example Problem Analysis** - Largest Rectangle in Histogram:

- Increasing stack ensures we can calculate rectangles efficiently
    
- When we see a smaller bar, calculate all possible rectangles ending here
    
- Width calculation uses the property of monotonic stack
    
- Sentinel value ensures all remaining bars are processed
    

## **9. Hash Maps**

### **9.1 Frequency Counting**

**Core Concept**: Use hash maps to count occurrences and find patterns based on frequencies.

**Key Insights**:

- **O(1) lookup/update**: Hash maps provide constant time operations
    
- **Frequency patterns**: Many problems reduce to frequency analysis
    
- **Grouping**: Group elements by some property (anagrams, etc.)
    
- **Space-time tradeoff**: Use extra space for faster lookup
    

**When to Use**:

- Counting occurrences of elements
    
- Finding duplicates or unique elements
    
- Grouping anagrams or similar strings
    
- Two sum and related problems
    
- Pattern matching in strings/arrays
    

**Template**:

```
def frequency_analysis(arr):
    freq_map = defaultdict(int)
    
    # Count frequencies
    for element in arr:
        freq_map[element] += 1
    
    # Process based on frequencies
    result = []
    for element, frequency in freq_map.items():
        if frequency > 1:  # or whatever condition
            result.append(element)
    
    return result
```

**Example Problem Analysis** - Group Anagrams:

- Key insight: Anagrams have same character frequencies
    
- Use sorted string as hash key for grouping
    
- All anagrams map to same key in hash table
    
- Time complexity: O(n × k log k) where k is average string length
    

### **9.2 Prefix Sum with HashMap**

**Core Concept**: Combine prefix sum technique with hash map for efficient subarray queries.

**Key Insights**:

- **Prefix sum**: sum[0...i] = sum[0...j] + sum[j+1...i]
    
- **Subarray sum**: sum[i...j] = prefix_sum[j] - prefix_sum[i-1]
    
- **Hash map usage**: Store prefix sums and their indices
    
- **Key insight**: If prefix_sum[j] - prefix_sum[i] = target, then subarray sum = target
    

**When to Use**:

- Subarray sum equals K
    
- Continuous subarray sum
    
- Binary subarrays with given sum
    
- Subarray with equal 0s and 1s
    

**Template**:

```
def subarray_sum_equals_k(nums, k):
    count = 0
    prefix_sum = 0
    sum_map = {0: 1}  # Initialize with sum 0 occurring once
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if (prefix_sum - k) in sum_map:
            count += sum_map[prefix_sum - k]
        
        # Update current prefix sum count
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count
```

**Example Problem Analysis** - Subarray Sum Equals K:

- If prefix_sum[j] - prefix_sum[i] = k, then subarray[i+1...j] has sum k
    
- Rearranging: prefix_sum[i] = prefix_sum[j] - k
    
- For each position j, check if (current_sum - k) was seen before
    
- Count how many times that difference occurred
    

## **10. Tree Breadth First Search (BFS)**

### **10.1 Level Order Traversal**

**Core Concept**: Process tree nodes level by level using a queue.

**Key Insights**:

- **Queue-based**: Use queue to maintain nodes in FIFO order
    
- **Level separation**: Process all nodes at current level before moving to next
    
- **Level size**: Track how many nodes are in current level
    
- **Applications**: Level-wise processing, finding tree properties
    

**When to Use**:

- Level order traversal problems
    
- Finding tree depth/height
    
- Connect nodes at same level
    
- Binary tree serialization/deserialization
    
- Finding rightmost/leftmost nodes at each level
    

**Template**:

```
from collections import deque

def level_order_traversal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        # Process all nodes in current level
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

**Example Problem Analysis** - Binary Tree Level Order Traversal:

- Use queue to maintain nodes to be processed
    
- Process one level at a time by tracking level size
    
- Add children of current level nodes for next iteration
    
- Each level becomes a separate list in result
    

### **10.2 Connect Nodes Pattern**

**Core Concept**: Connect nodes at the same level using BFS traversal properties.

**Key Insights**:

- **Level processing**: Process nodes level by level
    
- **Connection strategy**: Connect adjacent nodes in same level
    
- **Perfect vs complete**: Different strategies for different tree types
    
- **Space optimization**: Can be done in O(1) space for perfect binary trees
    

**When to Use**:

- Populating next right pointers
    
- Finding minimum depth of tree
    
- Level-wise node connections
    
- Tree serialization with level information
    

**Template**:

```
def connect_nodes(root):
    if not root:
        return root
    
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Connect to next node in same level
            if i < level_size - 1:
                node.next = queue[0]  # Next node in queue
            
            # Add children
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return root
```

**Example Problem Analysis** - Populating Next Right Pointers:

- Process nodes level by level using BFS
    
- For each node (except last in level), connect to next node in queue
    
- Queue naturally maintains left-to-right order within each level
    
- Can optimize for perfect binary trees using existing connections
    

## **11. Tree Depth First Search (DFS)**

### **11.1 Binary Tree Paths**

**Core Concept**: Use recursive DFS to explore all paths from root to leaves or specific targets.

**Key Insights**:

- **Recursive nature**: Tree problems naturally fit recursive solutions
    
- **Path tracking**: Maintain current path and backtrack when needed
    
- **Base cases**: Handle leaf nodes and null nodes appropriately
    
- **Path construction**: Build path incrementally and remove when backtracking
    

**When to Use**:

- Finding all root-to-leaf paths
    
- Path sum problems (specific sum or counting paths)
    
- Maximum/minimum path problems
    
- Tree path queries
    

```
Template (Path Sum):
def has_path_sum(root, target_sum):
    if not root:
        return False
    
    # Leaf node check
    if not root.left and not root.right:
        return root.val == target_sum
    
    # Recursive case: subtract current value and check children
    remaining_sum = target_sum - root.val
    return (has_path_sum(root.left, remaining_sum) or 
            has_path_sum(root.right, remaining_sum))

Template (All Paths):
def all_paths_sum(root, target_sum):
    def dfs(node, current_path, remaining_sum):
        if not node:
            return
        
        # Add current node to path
        current_path.append(node.val)
        
        # Check if leaf node with target sum
        if not node.left and not node.right and remaining_sum == node.val:
            result.append(current_path[:])  # Copy current path
        
        # Recurse on children
        dfs(node.left, current_path, remaining_sum - node.val)
        dfs(node.right, current_path, remaining_sum - node.val)
        
        # Backtrack
        current_path.pop()
    
    result = []
    dfs(root, [], target_sum)
    return result
```

**Example Problem Analysis** - Path Sum II:

- Use DFS to explore all possible paths
    
- Maintain current path and remaining sum
    
- When reaching leaf, check if remaining sum equals leaf value
    
- Backtrack by removing current node from path
    

### **11.2 Tree Diameter and Height**

**Core Concept**: Calculate tree properties using post-order DFS traversal.

**Key Insights**:

- **Post-order processing**: Process children before current node
    
- **Global vs local**: Distinguish between global answer and local contribution
    
- **Diameter calculation**: Max path through any node = left_height + right_height + 1
    
- **Return vs update**: Return local info, update global info
    

**When to Use**:

- Finding tree diameter (longest path between any two nodes)
    
- Calculating tree height/depth
    
- Checking if tree is balanced
    
- Any problem requiring tree structural analysis
    

### **12. Graphs**

Graphs are data structures that represent networks of nodes (vertices) and connections (edges). They're essential in modeling real-world systems like social networks, maps, and networks.

#### **12.1 Graph Traversal (DFS/BFS)**

**Core Concept:** Explore all vertices and edges of a graph using systematic traversal methods.

**Key Insights:**

- **DFS (Depth-First Search):** Dives deep along each path before backtracking. Implemented with recursion or a stack.
    
- **BFS (Breadth-First Search):** Explores neighbors level-by-level using a queue. Especially useful for shortest paths in unweighted graphs.
    
- **Visited tracking:** Use a visited set or boolean array to avoid cycles and redundant visits.
    
- **Graph representation:** Use adjacency list or matrix depending on density.
    

**When to Use:**

- Checking connectivity in graphs.
    
- Detecting cycles in directed/undirected graphs.
    
- Traversing components in disconnected graphs.
    
- Performing level-order or depth-order exploration.
    
- Finding shortest paths in **unweighted** graphs (BFS).
    

#### **12.2 Shortest Path Algorithms**

**Core Concept:** Compute minimum distances between nodes in a graph using optimized strategies.

**Key Insights:**

- **Dijkstra’s Algorithm:** Greedy algorithm using a min-heap to always pick the node with the current shortest distance. Works only with non-negative weights.
    
- **Bellman-Ford:** Relaxes all edges repeatedly. Can detect negative cycles. Slower but more flexible.
    
- **Floyd-Warshall:** Dynamic programming-based algorithm for all-pairs shortest paths. Useful in dense graphs.
    
- **BFS (for unweighted graphs):** Every edge has the same cost; BFS ensures the shortest path due to level-wise exploration.
    

**When to Use:**

- Computing shortest path from source to all nodes.
    
- Solving road networks, routing systems, or pathfinding in maps.
    
- Detecting negative cycles (Bellman-Ford).
    
- Finding pairwise shortest distances (Floyd-Warshall).
    

---

### **13. Island (Matrix traversal)**

Grid-based problems are often modeled as graphs where each cell is a node.

#### **13.1 Connected Components in Matrix**

**Core Concept:** Use DFS/BFS to identify all connected regions (or islands) in a matrix.

**Key Insights:**

- **Graph from grid:** Treat each cell as a node and adjacent cells (up, down, left, right) as edges.
    
- **DFS/BFS exploration:** Start from unvisited '1' cells and mark all connected '1's as visited.
    
- **Visited tracking:** Modify the grid in-place or use a separate visited matrix to prevent reprocessing.
    
- **Component counting:** Each DFS/BFS run from an unvisited land cell signifies a new component.
    

**When to Use:**

- Counting islands (e.g., Leetcode 200: Number of Islands).
    
- Labeling or grouping contiguous regions.
    
- Image segmentation or blob detection problems.
    
- Any problem involving clustered structure in 2D space.
    

#### **13.2 Flood Fill Pattern**

**Core Concept:** Spread a change from a starting point to all adjacent similar elements.

**Key Insights:**

- **Recursion or queue:** Use DFS or BFS starting at the initial pixel.
    
- **Color matching:** Only spread to neighbors with the original color.
    
- **In-place update:** Change cell value during visit to avoid revisiting.
    
- **Edge cases:** Handle early exit if target color equals new color.
    

**When to Use:**

- Repainting regions in matrix problems (e.g., Leetcode 733: Flood Fill).
    
- Connected coloring in game-like scenarios.
    
- Identifying and replacing zones or regions.
    
- Problems involving expansion from a seed point.
    

---

### **14. Two Heaps**

This pattern is useful when you need to keep track of the median of a dynamically updating data stream.

#### **14.1 Median Finding**

**Core Concept:** Use two heaps to maintain balance and calculate the running median in logarithmic time.

**Key Insights:**

- **Two-heap approach:** Maintain a max-heap for the smaller half and a min-heap for the larger half.
    
- **Balancing:** Ensure the heaps are balanced in size (differ by at most 1) to maintain the median in the middle.
    
- **Median retrieval:**
    
    - If both heaps are equal size, median is average of tops.
        
    - If unequal, it's the top of the larger heap.
        
- **Heap operations:** Insert into one heap and rebalance if needed to maintain the size invariant.
    

**When to Use:**

- Real-time median tracking (e.g., streaming data).
    
- Sliding window median problems.
    
- Any scenario requiring dynamic insertion and real-time median calculation.
    

---

### **15. Subsets**

Generating subsets (the power set) is fundamental in problems involving combinations.

#### **15.1 Generate All Subsets**

**Core Concept:** Use backtracking or bitmasking to explore all subset combinations.

**Key Insights:**

- **Backtracking approach:** At each index, decide to include or exclude the current element.
    
- **Recursive tree:** The recursion forms a binary tree where each node splits on include/exclude.
    
- **Bitmasking approach:** Iterate from 0 to 2^n - 1. The binary representation acts as a mask for inclusion.
    
- **Immutable sets:** Often collect results into a list of lists or sets to preserve state.
    

**When to Use:**

- Generating the power set of a collection.
    
- Solving problems involving choice combinations (e.g., subset sum).
    
- Filtering subsets with constraints.
    
- Preparation for permutation or combination generation problems.
    

---

### **16. Modified Binary Search**

Binary search is a powerful technique when the solution space is ordered.

#### **16.1 Search in Rotated Array**

**Core Concept:** Apply modified binary search to find an element in a rotated sorted array.

**Key Insights:**

- **Identify sorted half:** One half of the array is always sorted. Check mid vs left/right to determine.
    
- **Shrink search space:** Decide whether to search in the sorted half or the other based on the target’s position.
    
- **Binary search logic preserved:** Still relies on mid-point splitting and log(n) complexity.
    
- **Edge conditions:** Carefully handle equality and tight bounds when array contains duplicates.
    

**When to Use:**

- Searching in rotated arrays (e.g., Leetcode 33: Search in Rotated Sorted Array).
    
- Searching under transformation (shifted or circular sorted data).
    
- Efficient O(log n) search in modified sorted structures.
    

#### **16.2 Search in Answer Space**

**Core Concept:** Perform binary search on potential answers rather than positions or values directly.

**Key Insights:**

- **Feasibility check function:** Convert problem into a boolean function: can we achieve this answer?
    
- **Search bounds:** Define low and high based on the smallest and largest possible valid answers.
    
- **Optimization problems:** Frequently used in problems asking for "minimum x that satisfies y" or "maximum x under constraint z".
    
- **Monotonicity is key:** The solution space must be monotonic (true/false must follow a contiguous block).
    

**When to Use:**

- Allocation and scheduling problems (e.g., minimum max load, shipping days).
    
- Maximize/minimize under constraints (e.g., Split Array Largest Sum).
    
- Problems involving resource distribution.
    

---

### **17. Bitwise XOR**

XOR has unique properties that make it useful in problems involving parity or cancellation.

#### **17.1 Single Number Pattern**

**Core Concept:** Use XOR to identify unique elements in a list where others appear in pairs.

**Key Insights:**

- **XOR property:** `a ^ a = 0` and `0 ^ b = b`. This cancels out duplicates and isolates the unique value.
    
- **Associative and commutative:** XOR can be applied in any order, which makes it efficient for linear scans.
    
- **Constant space:** Only one variable needed to track result.
    
- **Extension:** Can be adapted to find two unique elements using XOR and bit manipulation.
    

**When to Use:**

- Finding a single non-repeating number where others occur twice.
    
- Finding two unique numbers with all others in pairs.
    
- Problems involving parity-based cancellation.
    

---

### **18. Top 'K' Elements**

This pattern efficiently finds the top K largest or smallest elements in a collection.

#### **18.1 Kth Largest/Smallest**

**Core Concept:** Use a heap (priority queue) to track the top K elements efficiently.

**Key Insights:**

- **Min-heap for K largest:** Keep a min-heap of size K, discard smaller elements.
    
- **Max-heap for K smallest:** Keep a max-heap of size K, discard larger elements.
    
- **Heap size invariant:** Always maintain K elements in the heap to ensure efficiency.
    
- **Partial sorting:** Avoid full sort; only maintain K relevant items.
    

**When to Use:**

- Finding Kth largest or smallest values in a stream or array.
    
- Keeping track of top or bottom K elements dynamically.
    
- Problems where full sorting is inefficient (O(n log k) instead of O(n log n)).
    

---

### **19. K-way Merge**

#### **19.1 Merge Sorted Lists/Arrays**

**Core Concept:** Use a min-heap to merge multiple sorted lists or arrays into a single sorted structure.

**Key Insights:**

- **Min-heap usage:** Push the first element of each list into the heap.
    
- **Efficient comparison:** Always extract the smallest element and push its next from the same list.
    
- **Tuple storage:** Store value, list index, and element index in heap to track source.
    
- **Termination:** Continue until the heap is empty, guaranteeing all elements are merged.
    

**When to Use:**

- Merging K sorted linked lists or arrays (e.g., Leetcode 23).
    
- External sorting with limited memory.
    
- Streaming multiple sorted input streams.
    

---

### **20. Greedy Algorithms**

#### **20.1 Activity Selection Pattern**

**Core Concept:** Select the maximum number of non-overlapping intervals by greedily choosing the earliest finishing ones.

**Key Insights:**

- **Sort by end time:** Optimal substructure is maintained by choosing intervals with the smallest finish time.
    
- **Greedy choice:** Once an activity is selected, skip all overlapping ones.
    
- **No backtracking needed:** Greedy strategy leads to optimal solution.
    

**When to Use:**

- Scheduling maximum jobs or meetings.
    
- Choosing compatible intervals.
    
- Solving classic interval scheduling or job sequencing problems.
    

To select the maximum number of non-overlapping intervals, sort the intervals by their end time. Then iterate and pick the next interval that starts after the last picked one ends.

#### **20.2 String Construction Greedy**

**Core Concept:** Use greedy decisions (like stack or priority) to build optimal strings under constraints.

**Key Insights:**

- **Lexicographic control:** Remove characters to maintain smallest/largest possible result.
    
- **Stack-based building:** Push characters and pop larger ones when constraints allow (e.g., k removals).
    
- **Character frequency tracking:** Use counters when constraints include character limits.
    

**When to Use:**

- Removing characters to build smallest string (e.g., Leetcode 402: Remove K Digits).
    
- Greedy character inclusion/exclusion problems.
    
- Constructing valid strings under lexicographical or structural rules.
    

---

### **21. 0/1 Knapsack (Dynamic Programming)**

#### **21.1 Classic Knapsack Variations**

**Core Concept:** Solve optimization problems by making binary (yes/no) choices under constraints using dynamic programming.

**Key Insights:**

- **DP table definition:** Use `dp[i][w]` to represent the max value using first i items with capacity w.
    
- **Choice logic:** At each item, choose to either include it (`value + dp[i-1][w-weight]`) or exclude it (`dp[i-1][w]`).
    
- **1D optimization:** Space can be reduced to 1D array updated in reverse.
    
- **Variations:** Subset sum (boolean DP), unbounded knapsack (allow multiple uses of same item), bounded (limited item count).
    

**When to Use:**

- Problems with a fixed resource (e.g., weight, budget).
    
- Maximizing/minimizing value given item constraints.
    
- Finding subsets that match specific criteria.
    
- Decision-making under limited capacity.
    

---

### **22. Backtracking**

#### **22.1 Combination Generation**

**Core Concept:** Recursively build partial solutions and backtrack when constraints are violated.

**Key Insights:**

- **Decision tree:** For each element, explore both inclusion and exclusion.
    
- **Prune early:** If a partial path can’t lead to a valid solution, return early.
    
- **Recursive template:** Function tracks current index, path, and base case.
    
- **Stack/Path usage:** Maintain current path of elements and backtrack by removing last element after recursive call.
    

**When to Use:**

- Generating combinations/permutations/subsets.
    
- Problems with constraints that must be validated along the way.
    
- Constructing valid outputs like parenthesis, combinations, or sum groups.
    

#### **22.1 Combination Generation**

Used to generate all valid combinations of elements. For example, combinations of `k` elements from a set of `n`, or generating valid parentheses. Backtracking explores options, prunes invalid ones, and proceeds recursively.

#### **22.2 N-Queens and Board Problems**

**Core Concept:** Explore board configurations recursively while enforcing spatial constraints.

**Key Insights:**

- **Board traversal:** Place pieces row by row while maintaining constraints.
    
- **Constraint tracking:** Use sets or arrays to track used columns, diagonals.
    
- **Recursive placement:** Try placing a queen, recurse for next row, and backtrack on failure.
    
- **Prune efficiently:** Return early if current column/diagonal is unsafe.
    

**When to Use:**

- Solving classic board placement problems (N-Queens, Sudoku).
    
- Placing tokens/markers with spatial restrictions.
    
- Constructing layouts without overlap/conflict.
    

---

### **23. Trie**

A Trie is a tree-like data structure used for efficient retrieval of strings.

#### **23.1 Prefix Tree Operations**

**Core Concept:** Use a trie (prefix tree) to efficiently perform operations on prefixes of strings.

**Key Insights:**

- **Character-wise storage:** Each node represents one character in a word; child links represent possible next characters.
    
- **Prefix querying:** Easily check if a prefix exists without scanning all words.
    
- **Insertion and search time:** O(L) where L is the length of the word or prefix.
    
- **Space usage:** Uses more memory than hash sets due to tree structure, but benefits from prefix sharing.
    

**When to Use:**

- Autocomplete systems or prefix-based word searches.
    
- Dictionary implementations where prefix checking is frequent.
    
- Problems requiring fast lookups for startsWith or whole word existence.
    
- Efficient implementation of word games like Boggle or word squares.
    

---

### **24. Topological Sort (Graph)**

Used to order vertices in a Directed Acyclic Graph (DAG) such that for every directed edge u -> v, u comes before v.

#### **24.1 Course Prerequisites**

**Core Concept:** Use topological sorting on a Directed Acyclic Graph (DAG) to resolve dependencies.

**Key Insights:**

- **DFS-based approach:** Track visited and recursion stack to detect cycles and order nodes post-order.
    
- **Kahn’s Algorithm:** Use in-degree and queue to order nodes with no incoming edges.
    
- **Cycle detection:** If sorting doesn't include all nodes, a cycle exists.
    
- **Multiple valid orders:** Topological sort may yield more than one valid result.
    

**When to Use:**

- Resolving dependencies (e.g., course scheduling, build systems).
    
- Ordering tasks with prerequisites.
    
- DAG traversal where order of execution matters.
    

---

### **25. Union Find**

#### **25.1 Connected Components**

**Core Concept:** Use Disjoint Set Union (DSU) to manage dynamic connectivity in a network of elements.

**Key Insights:**

- **Find and Union:** `find(x)` returns root of set x; `union(x, y)` merges the sets.
    
- **Path compression:** Speeds up find operation by flattening the tree structure.
    
- **Union by rank/size:** Ensures smaller trees are merged under larger trees to optimize depth.
    
- **Cycle detection:** Can determine if adding an edge creates a cycle in an undirected graph.
    

**When to Use:**

- Keeping track of connected components in dynamic graphs.
    
- Kruskal’s algorithm for Minimum Spanning Tree.
    
- Network connectivity problems.
    
- Grouping and equivalence relations.
    

---

### **26. Ordered Set**

#### **26.1 Range Queries with TreeMap/TreeSet**

**Core Concept:** Use balanced BST structures like TreeMap/TreeSet to maintain sorted elements and answer range-based queries efficiently.

**Key Insights:**

- **Logarithmic operations:** Supports O(log n) insert, delete, ceiling, floor, and range queries.
    
- **Sliding window:** Efficiently track values in a window with automatic ordering.
    
- **Duplicates:** TreeMap allows counting/frequency tracking; TreeSet holds unique sorted values.
    
- **Alternative to heaps:** Offers more flexibility in range queries with ordering preserved.
    

**When to Use:**

- Maintaining ordered elements for real-time range queries.
    
- Problems requiring nearest smaller/larger elements.
    
- Dynamic window median or max/min.
    
- Efficient handling of intervals or sorted sequences.
    

---

### **27. Multi-thread**

#### **27.1 Thread Synchronization**

**Core Concept:** Coordinate access to shared resources across threads to prevent race conditions and ensure consistent behavior.

**Key Insights:**

- **Locks and semaphores:** Use mutual exclusion (mutex) to protect critical sections.
    
- **Barriers and countdown latches:** Coordinate the start or end of thread groups.
    
- **Condition variables:** Let threads wait for specific states before proceeding.
    
- **Deadlocks and livelocks:** Must design carefully to avoid these synchronization traps.
    

**When to Use:**

- Designing thread-safe shared resources.
    
- Sequencing thread execution (e.g., print FooBar or numbers in order).
    
- Solving concurrency problems like producer-consumer, dining philosophers, H2O formation.
    
- Systems where multiple threads access shared memory or perform coordinated tasks.
    

---

### **28. Miscellaneous Advanced Patterns**

#### **28.1 Line Sweep Algorithm**

**Core Concept:** Sort and process events in one dimension to track active intervals or conditions.

**Key Insights:**

- **Event-based:** Convert intervals into start/end events.
    
- **Sorted order:** Process events in increasing x-coordinate (or time).
    
- **Active structure:** Maintain a data structure (multiset, heap) of active intervals.
    
- **Efficient updates:** Add/remove intervals as the sweep progresses.
    

**When to Use:**

- Meeting room or interval scheduling problems.
    
- Counting overlaps or maximum concurrent events.
    
- The skyline problem.
    

#### **28.2 Coordinate Compression**

**Core Concept:** Map large, sparse or non-continuous values into a smaller range while preserving order.

**Key Insights:**

- **Sorting and mapping:** Sort unique values and map each to its index.
    
- **Preserve order:** The compressed values retain the original ordering.
    
- **Indexing benefits:** Allows use of arrays or trees on originally large values.
    

**When to Use:**

- Range-based algorithms with large coordinate ranges.
    
- Segment Tree, Fenwick Tree (BIT) problems.
    
- Memory-efficient solutions in time-sensitive environments.
    

#### **28.3 Segment Tree / Fenwick Tree**

**Core Concept:** Enable efficient range queries and updates using tree-like data structures.

**Key Insights:**

- **Segment Tree:** Recursively divides range into segments; supports min, max, sum, etc.
    
- **Fenwick Tree (BIT):** Efficient prefix sums; simpler and more compact for sum operations.
    
- **Lazy propagation:** Used in Segment Tree to delay updates until necessary.
    
- **Time complexity:** O(log n) for both query and update.
    

**When to Use:**

- Frequent range sum/min/max queries and point updates.
    
- Dynamic arrays with mutable elements.
    
- Problems requiring both updates and queries in logarithmic time.
    

#### **28.4 Manacher's Algorithm**

**Core Concept:** Find the longest palindromic substring in linear time using symmetry and center expansion.

**Key Insights:**

- **String transformation:** Insert separators to handle odd/even-length uniformly.
    
- **Center expansion:** Track center and right boundary of known palindromes.
    
- **Mirror property:** Use already computed results to skip redundant checks.
    
- **Preprocessing:** Converts O(n^2) brute-force to O(n).
    

**When to Use:**

- Problems involving longest palindromic substring.
    
- Efficient alternative to center-expansion or dynamic programming.
    
- Competitive programming or palindrome-heavy problems.
    

#### **28.5 KMP Algorithm (String Matching)**

**Core Concept:** Use prefix table to skip unnecessary comparisons during string matching.

**Key Insights:**

- **Prefix function:** Precompute longest prefix-suffix array (LPS).
    
- **Mismatch handling:** Use LPS to jump to next valid match location.
    
- **O(n + m):** Linear time pattern matching.
    
- **Avoids re-checking:** No backtracking in the text.
    

**When to Use:**

- Substring search problems.
    
- DNA sequence matching, plagiarism detection.
    
- Efficient repeated pattern detection.
    

#### **29.1 State Machine DP**

**Core Concept:** Model problems as a sequence of state transitions and use DP to optimize across those states.

**Key Insights:**

- **Define states:** Explicitly represent different modes (e.g., hold/sell/cooldown in stock problems).
    
- **State transitions:** Build recurrence relations for valid moves between states.
    
- **Space optimization:** Use rolling variables for space-efficient implementation.
    
- **Trace optimal path:** Track the state transitions to reconstruct decisions if needed.
    

**When to Use:**

- Problems involving toggling states (buy/sell, on/off, locked/unlocked).
    
- Dynamic programming with memory of previous states.
    
- Financial scenarios like stock trading with constraints.
    

#### **29.2 Interval DP**

**Core Concept:** Solve optimization problems over ranges by dividing into smaller subintervals.

**Key Insights:**

- **DP[i][j]:** Represents optimal result over interval (i, j).
    
- **Divide and conquer:** Try all possible splits `k` between i and j and combine solutions.
    
- **Overlapping subintervals:** Compute results in increasing order of interval length.
    
- **Memoization helps:** Store overlapping results for repeated reuse.
    

**When to Use:**

- Problems like matrix chain multiplication, palindrome partitioning.
    
- Dynamic programming over sequences that must be broken into parts.
    
- Parsing or evaluating expressions with variable operator placement.
    

#### **29.3 Digit DP**

**Core Concept:** Count numbers satisfying constraints by examining digits recursively with memoization.

**Key Insights:**

- **Recursive position-wise DP:** At each digit, decide which digit to place next based on tight bounds.
    
- **Tightness and leading zero flags:** Used to restrict search space.
    
- **Memoization:** Cache results based on position, tight flag, and accumulated state.
    
- **Base cases:** Clear exit conditions for valid/invalid states.
    

**When to Use:**

- Counting numbers with digit constraints (e.g., no repeating digits, sum of digits).
    
- Problems with numerical upper/lower bounds.
    
- Range-based integer counting questions.
    

#### **29.4 Tree DP**

**Core Concept:** Perform bottom-up dynamic programming on tree structures using post-order traversal.

**Key Insights:**

- **Post-order traversal:** Solve subtrees first, then aggregate results upward.
    
- **Multiple return values:** Store multiple values (e.g., include/exclude node).
    
- **Subtree combination:** Use results from child subtrees to compute current result.
    
- **Avoid recomputation:** Cache intermediate results where applicable.
    

**When to Use:**

- Aggregation problems on trees (e.g., sum, max weight independent set).
    
- Subtree-based inclusion/exclusion logic.
    
- Recursive structural optimization in trees.
    

---

### **30. Game Theory Patterns**

#### **30.1 Minimax Algorithm**

**Core Concept:** Simulate a two-player game using recursive backtracking and optimize both players' moves.

**Key Insights:**

- **Recursive game tree:** Max player tries to maximize score, Min player tries to minimize.
    
- **Terminal conditions:** Define end-game and evaluation function.
    
- **Alpha-beta pruning:** Cut off branches that can't affect final decision to optimize runtime.
    
- **Turn tracking:** Use depth or boolean flag to alternate player moves.
    

**When to Use:**

- Two-player games (e.g., Tic-Tac-Toe, Chess endgames).
    
- Turn-based decision simulations.
    
- Game strategy exploration with optimal outcomes.
    

---

### **31. Mathematical Patterns**

#### **31.1 Number Theory**

**Core Concept:** Use mathematical principles to analyze and solve numerical problems.

**Key Insights:**

- **GCD/LCM:** Greatest Common Divisor and Least Common Multiple are foundational tools.
    
- **Modulo arithmetic:** Essential for handling large numbers.
    
- **Sieve of Eratosthenes:** Efficient prime generation.
    
- **Euler’s/Fermat’s Theorem:** Modular inverses and power reduction.
    

**When to Use:**

- Problems involving divisibility, remainders, or number properties.
    
- Modular arithmetic problems (e.g., modular exponentiation).
    
- Prime factorization, GCD queries, modular inverses.
    

#### **31.2 Combinatorics**

**Core Concept:** Count arrangements and selections using mathematical formulas and principles.

**Key Insights:**

- **Permutations and combinations:** Use nCr, nPr, and factorial formulas.
    
- **Pascal’s Triangle:** Efficient method to compute combinations.
    
- **Inclusion-Exclusion:** Handle overlaps in counting problems.
    
- **DP-based counting:** For cases where analytical formulas aren't applicable.
    

**When to Use:**

- Counting subsets, permutations, and arrangements.
    
- Problems involving probability or constraints in selection.
    
- Dynamic programming versions of path-counting or string interleaving.
    

---

### **32. Design Patterns**

#### **32.1 LRU and Cache Design**

**Core Concept:** Design efficient caching mechanisms using a combination of data structures.

**Key Insights:**

- **HashMap + Doubly Linked List:** HashMap provides O(1) access, list maintains order.
    
- **Eviction policy:** Remove least recently used item when capacity is exceeded.
    
- **Update on access:** Move accessed item to the front.
    
- **Thread-safe variants:** Often paired with locks in real-world systems.
    

**When to Use:**

- Implementing LRU caches, database caching layers.
    
- Page replacement algorithms.
    
- Problems requiring fixed-size recent-access tracking.
    

#### **32.2 Iterator Design**

**Core Concept:** Build custom iterators over complex data structures for controlled traversal.

**Key Insights:**

- **hasNext() / next():** Implement standard iterator interface.
    
- **Flattening structures:** Use stacks or queues to traverse nested/2D data.
    
- **Pre-processing vs lazy loading:** Choose based on space vs performance tradeoff.
    
- **Encapsulation:** Hide underlying structure while exposing consistent traversal.
    

**When to Use:**

- Navigating nested data (e.g., NestedInteger, 2D iterators).
    
- Streamlined sequential access to complex structures.
    
- Building custom iteration logic for user-facing libraries.
    

---Create custom iterators over complex structures. Examples: 2D vector iterator, nested list iterator. Implement `hasNext()` and `next()` methods carefully.

---

# **Complete LeetCode Sub-Patterns Guide with Practice Problems**

## **1. Two Pointers Patterns**

### **1.1 Opposite Direction Two Pointers**

**Use Case:** When you need to search from both ends of an array

- [1. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
    
- [15. 3Sum](https://leetcode.com/problems/3sum/)
    
- [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
    
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
    
- [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
    

### **1.2 Same Direction Two Pointers (Fast-Slow)**

**Use Case:** When you need different speeds of iteration

- [26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
    
- [27. Remove Element](https://leetcode.com/problems/remove-element/)
    
- [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)
    
- [80. Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)
    
- [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
    

### **1.3 Three Pointers**

**Use Case:** For problems requiring three elements or partitioning

- [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/)
    
- [18. 4Sum](https://leetcode.com/problems/4sum/)
    
- [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
    
- [259. 3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)
    
- [Dutch Flag Problem - Sort Colors](https://leetcode.com/problems/sort-colors/)
    

## **2. Fast & Slow Pointers (Floyd's Cycle Detection)**

### **2.1 Cycle Detection in Linked Lists**

**Use Case:** Detecting cycles in linked structures

- [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
    
- [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
    
- [202. Happy Number](https://leetcode.com/problems/happy-number/)
    
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
    
- [457. Circular Array Loop](https://leetcode.com/problems/circular-array-loop/)
    

### **2.2 Finding Middle Element**

**Use Case:** Finding middle or specific position in one pass

- [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
    
- [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
    
- [143. Reorder List](https://leetcode.com/problems/reorder-list/)
    
- [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
    
- [61. Rotate List](https://leetcode.com/problems/rotate-list/)
    

## **3. Sliding Window Patterns**

### **3.1 Fixed Size Sliding Window**

**Use Case:** Window of fixed size k

- [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
    
- [1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)
    
- [1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)
    
- [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)
    
- [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
    

### **3.2 Variable Size Sliding Window - Maximum**

**Use Case:** Find maximum window satisfying condition

- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
    
- [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
    
- [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)
    
- [1838. Frequency of the Most Frequent Element](https://leetcode.com/problems/frequency-of-the-most-frequent-element/)
    
- [340. Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
    

### **3.3 Variable Size Sliding Window - Minimum**

**Use Case:** Find minimum window satisfying condition

- [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
    
- [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
    
- [862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)
    
- [727. Minimum Window Subsequence](https://leetcode.com/problems/minimum-window-subsequence/)
    
- [1234. Replace the Substring for Balanced String](https://leetcode.com/problems/replace-the-substring-for-balanced-string/)
    

## **4. Merge Intervals**

### **4.1 Overlapping Intervals**

**Use Case:** Merge or handle overlapping intervals

- [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
    
- [57. Insert Interval](https://leetcode.com/problems/insert-interval/)
    
- [435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
    
- [452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)
    
- [1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/)
    

### **4.2 Meeting Rooms Pattern**

**Use Case:** Scheduling and resource allocation

- [252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
    
- [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
    
- [1094. Car Pooling](https://leetcode.com/problems/car-pooling/)
    
- [732. My Calendar III](https://leetcode.com/problems/my-calendar-iii/)
    
- [729. My Calendar I](https://leetcode.com/problems/my-calendar-i/)
    

## **5. Cyclic Sort**

### **5.1 Missing Number Pattern**

**Use Case:** Find missing elements in range [1,n] or [0,n]

- [268. Missing Number](https://leetcode.com/problems/missing-number/)
    
- [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
    
- [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
    
- [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
    
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    

## **6. In-place Reversal of LinkedList**

### **6.1 Basic Reversal**

**Use Case:** Reverse linked list or parts of it

- [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
    
- [92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
    
- [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
    
- [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
    
- [61. Rotate List](https://leetcode.com/problems/rotate-list/)
    

## **7. Stack Patterns**

### **7.1 Basic Stack Operations**

**Use Case:** Parentheses, brackets, and matching problems

- [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
    
- [155. Min Stack](https://leetcode.com/problems/min-stack/)
    
- [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
    
- [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)
    
- [1047. Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)
    

### **7.2 Expression Evaluation**

**Use Case:** Evaluate mathematical expressions

- [150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
    
- [224. Basic Calculator](https://leetcode.com/problems/basic-calculator/)
    
- [227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)
    
- [772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)
    
- [394. Decode String](https://leetcode.com/problems/decode-string/)
    

## **8. Monotonic Stack**

### **8.1 Next Greater/Smaller Element**

**Use Case:** Find next greater or smaller elements

- [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
    
- [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
    
- [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
    
- [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/)
    
- [1019. Next Greater Node In Linked List](https://leetcode.com/problems/next-greater-node-in-linked-list/)
    

### **8.2 Histogram Pattern**

**Use Case:** Rectangle area problems

- [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
    
- [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
    
- [1793. Maximum Score of a Good Subarray](https://leetcode.com/problems/maximum-score-of-a-good-subarray/)
    
- [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)
    
- [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
    

## **9. Hash Maps**

### **9.1 Frequency Counting**

**Use Case:** Count occurrences and find patterns

- [1. Two Sum](https://leetcode.com/problems/two-sum/)
    
- [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)
    
- [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)
    
- [447. Number of Boomerangs](https://leetcode.com/problems/number-of-boomerangs/)
    
- [219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)
    

### **9.2 Prefix Sum with HashMap**

**Use Case:** Subarray sum problems

- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
    
- [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
    
- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
    
- [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/)
    
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
    

## **10. Tree Breadth First Search (BFS)**

### **10.1 Level Order Traversal**

**Use Case:** Process tree level by level

- [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
    
- [107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
    
- [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
    
- [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
    
- [515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)
    

### **10.2 Connect Nodes Pattern**

**Use Case:** Connect nodes at same level

- [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
    
- [117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)
    
- [111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
    
- [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
    
- [637. Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)
    

## **11. Tree Depth First Search (DFS)**

### **11.1 Binary Tree Paths**

**Use Case:** Find all paths or specific paths

- [112. Path Sum](https://leetcode.com/problems/path-sum/)
    
- [113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
    
- [257. Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)
    
- [437. Path Sum III](https://leetcode.com/problems/path-sum-iii/)
    
- [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
    

### **can**

## **Practice Strategy**

### **Beginner Level (Start Here)**

1. **Two Pointers (Opposite Direction)** - Master the basics
    
2. **Sliding Window (Fixed Size)** - Learn window management
    
3. **Hash Maps (Frequency Counting)** - Understand key-value relationships
    
4. **Stack (Basic Operations)** - Practice LIFO operations
    
5. **Tree BFS (Level Order)** - Learn tree traversal
    

### **Intermediate Level**

1. **Dynamic Programming (State Machine)** - Build DP intuition
    
2. **Graph Traversal** - Master DFS and BFS on graphs
    
3. **Binary Search (Modified)** - Advanced search techniques
    
4. **Backtracking** - Learn systematic exploration
    
5. **Union Find** - Understand connectivity problems
    

### **Advanced Level**

1. **Segment Tree / Fenwick Tree** - Range query optimization
    
2. **Line Sweep Algorithm** - Event processing
    
3. **Game Theory** - Strategic decision making
    
4. **Advanced DP (Interval, Digit)** - Complex state transitions
    
5. **String Algorithms (KMP, Manacher)** - Efficient string processing
    

## **Tips for Success**

1. **Start with easier problems** in each pattern before moving to harder ones
    
2. **Focus on understanding the pattern** rather than memorizing solutions
    
3. **Practice similar problems** consecutively to reinforce the pattern
    
4. **Time yourself** - aim for 30-45 minutes per problem initially
    
5. **Review and optimize** your solutions after solving
    
6. **Implement from scratch** without looking at solutions first
    
7. **Discuss solutions** with others to gain different perspectives
    

Remember: Consistency is key. Practice 1-2 problems daily rather than cramming many problems in one session.

# 14 WEEK ROADMAP

# WEEK 1

1. Introduction
    
2. Pair with Target Sum (easy) [LeetCode](https://leetcode.com/problems/two-sum/)
    
3. Remove Duplicates (easy) [LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-list/) [LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/) [LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/) [LeetCode](https://leetcode.com/problems/find-the-duplicate-number/) [LeetCode](https://leetcode.com/problems/duplicate-zeros/)
    
4. Squaring a Sorted Array (easy) [LeetCode](https://leetcode.com/problems/squares-of-a-sorted-array/)
    
5. Triplet Sum to Zero (medium) [LeetCode](https://leetcode.com/problems/3sum/)
    
6. Triplet Sum Close to Target (medium) [LeetCode](https://leetcode.com/problems/3sum-closest/)
    
7. Triplets with Smaller Sum (medium) [LintCode](https://www.lintcode.com/problem/3sum-smaller/description)
    
8. Subarrays with Product Less than a Target (medium) [LeetCode](https://leetcode.com/problems/subarray-product-less-than-k/)
    
9. Dutch National Flag Problem (medium) [CoderByte](https://coderbyte.com/algorithm/dutch-national-flag-sorting-problem)
    
10. Problem Challenge 1: Quadruple Sum to Target (medium) [Leetcode](https://leetcode.com/problems/4sum/)
    
11. Problem Challenge 2: Comparing Strings containing Backspaces (medium) [Leetcode](https://leetcode.com/problems/backspace-string-compare/)
    
12. Problem Challenge 3: Minimum Window Sort (medium) [Leetcode](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/) [Ideserve](https://www.ideserve.co.in/learn/minimum-length-subarray-sorting-which-results-in-sorted-array)
    
13. Introduction [emre.me](https://emre.me/coding-patterns/fast-slow-pointers/)
    
14. LinkedList Cycle (easy) [Leetcode](https://leetcode.com/problems/linked-list-cycle/)
    
15. Start of LinkedList Cycle (medium) [Leetcode](https://leetcode.com/problems/linked-list-cycle-ii/)
    
16. Happy Number (medium) [Leetcode](https://leetcode.com/problems/happy-number/)
    
17. Middle of the LinkedList (easy) [Leetcode](https://leetcode.com/problems/middle-of-the-linked-list/)
    
18. Problem Challenge 1: Palindrome LinkedList (medium) [Leetcode](https://leetcode.com/problems/palindrome-linked-list/)
    
19. Problem Challenge 2: Rearrange a LinkedList (medium) [Leetcode](https://leetcode.com/problems/reorder-list/)
    
20. Problem Challenge 3: Cycle in a Circular Array (hard) [Leetcode](https://leetcode.com/problems/circular-array-loop/)
    

## **WEEK 2: Sliding Window and Merge Intervals**

1. Introduction
    
2. Maximum Sum Subarray of Size K (easy)
    
3. Smallest Subarray with a given sum (easy) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/7XMlMEQPnnQ)
    
4. Longest Substring with K Distinct Characters (medium) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/YQQwQMWLx80)
    
5. Fruits into Baskets (medium) [LeetCode](https://leetcode.com/problems/fruit-into-baskets/)
    
6. No-repeat Substring (hard) [LeetCode](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
    
7. Longest Substring with Same Letters after Replacement (hard) [LeetCode](https://leetcode.com/problems/longest-repeating-character-replacement/)
    
8. Longest Subarray with Ones after Replacement (hard) [LeetCode](https://leetcode.com/problems/max-consecutive-ones-iii/)
    
9. Problem Challenge 1: Permutation in a String (hard) [Leetcode](https://leetcode.com/problems/permutation-in-string/)
    
10. Problem Challenge 2: String Anagrams (hard) [Leetcode](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
    
11. Problem Challenge 3: Smallest Window containing Substring (hard) [Leetcode](https://leetcode.com/problems/minimum-window-substring/)
    
12. Problem Challenge 4: Words Concatenation (hard) [Leetcode](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)
    
13. Introduction [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/3YVYvogqXpA)
    
14. Merge Intervals (medium) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/3jyVPKRA8yx)
    
15. Insert Interval (medium) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/3jKlyNMJPEM)
    
16. Intervals Intersection (medium) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/JExVVqRAN9D)
    
17. Conflicting Appointments (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/check-if-any-two-intervals-overlap-among-a-given-set-of-intervals/)
    
18. Problem Challenge 1: Minimum Meeting Rooms (hard) [Lintcode](https://www.lintcode.com/problem/meeting-rooms-ii/)
    
19. Problem Challenge 2: Maximum CPU Load (hard) [Geeksforgeeks](https://www.geeksforgeeks.org/maximum-cpu-load-from-the-given-list-of-jobs/)
    
20. Problem Challenge 3: Employee Free Time (hard) [CoderTrain](https://www.codertrain.co/employee-free-time)
    

## **WEEK 3: Cyclic Sort and In-place reversal of Linked List**

1. Introduction [emre.me](https://emre.me/coding-patterns/cyclic-sort/)
    
2. Cyclic Sort (easy) [Geeksforgeeks](https://www.geeksforgeeks.org/sort-an-array-which-contain-1-to-n-values-in-on-using-cycle-sort/)
    
3. Find the Missing Number (easy) [Leetcode](https://leetcode.com/problems/missing-number/)
    
4. Find all Missing Numbers (easy) [Leetcode](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)
    
5. Find the Duplicate Number (easy) [Leetcode](https://leetcode.com/problems/find-the-duplicate-number/)
    
6. Find all Duplicate Numbers (easy) [Leetcode](https://leetcode.com/problems/find-all-duplicates-in-an-array/)
    
7. Problem Challenge 1: Find the Corrupt Pair (easy) [TheCodingSimplified](https://thecodingsimplified.com/find-currupt-pair/)
    
8. Problem Challenge 2: Find the Smallest Missing Positive Number (medium) [Leetcode](https://leetcode.com/problems/first-missing-positive/)
    
9. Problem Challenge 3: Find the First K Missing Positive Numbers (hard) [TheCodingSimplified](https://thecodingsimplified.com/find-the-first-k-missing-positive-number/)
    
10. Introduction [emre.me](https://emre.me/coding-patterns/in-place-reversal-of-a-linked-list/)
    
11. Reverse a LinkedList (easy) [Leetcode](https://leetcode.com/problems/reverse-linked-list/)
    
12. Reverse a Sub-list (medium) [Leetcode](https://leetcode.com/problems/reverse-linked-list-ii/)
    
13. Reverse every K-element Sub-list (medium) [Leetcode](https://leetcode.com/problems/reverse-nodes-in-k-group/)
    
14. Problem Challenge 1: Reverse alternating K-element Sub-list (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/reverse-alternate-k-nodes-in-a-singly-linked-list/)
    
15. Problem Challenge 2: Rotate a LinkedList (medium) [Leetcode](https://leetcode.com/problems/rotate-list/)
    

## **WEEK 4: Stack and Monotonic Stack**

1. Introduction to Stack (Operations, Implementation, Applications)
    
2. Balanced Parentheses [Leetcode](https://leetcode.com/problems/valid-parentheses/description/)
    
3. Reverse a String
    
4. Decimal to Binary Conversion
    
5. Next Greater Element [Leetcode - I](https://leetcode.com/problems/next-greater-element-i/) [Leetcode -II](https://leetcode.com/problems/next-greater-element-ii/) [Leetcode - III (Hard)](https://leetcode.com/problems/next-greater-element-iv/)
    
6. Sorting a Stack
    
7. Simplify Path [Leetcode](https://leetcode.com/problems/simplify-path/)
    
8. Introduction to Monotonic Stack
    
9. Next Greater Element (easy) [Leetcode - I](https://leetcode.com/problems/next-greater-element-i/) [Leetcode -II](https://leetcode.com/problems/next-greater-element-ii/) [Leetcode - III (Hard)](https://leetcode.com/problems/next-greater-element-iv/)
    
10. Daily Temperatures (easy) [Leetcode](https://leetcode.com/problems/daily-temperatures/)
    
11. Remove Nodes From Linked List (easy) [Leetcode](https://leetcode.com/problems/remove-nodes-from-linked-list/)
    
12. Remove All Adjacent Duplicates In String (easy) [Leetcode](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)
    
13. Remove All Adjacent Duplicates in String II (medium) [Leetcode](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)
    
14. Remove K Digits (hard) [Leetcode](https://leetcode.com/problems/remove-k-digits/)
    

## **WEEK 5: Hash Maps and Tree : BFS**

1. Introduction (Hashing, Hash Tables, Issues)
    
2. First Non-repeating Character (easy) [Leetcode](https://leetcode.com/problems/first-unique-character-in-a-string/)
    
3. Largest Unique Number (easy) [Leetcode+](https://leetcode.com/problems/largest-unique-number/)
    
4. Maximum Number of Balloons (easy) [Leetcode](https://leetcode.com/problems/maximum-number-of-balloons/)
    
5. Longest Palindrome(easy) [Leetcode](https://leetcode.com/problems/longest-palindrome/)
    
6. Ransom Note (easy) [Leetcode](https://leetcode.com/problems/ransom-note/)
    
7. Binary Tree Level Order Traversal (easy) [Leetcode](https://leetcode.com/problems/binary-tree-level-order-traversal/)
    
8. Reverse Level Order Traversal (easy) [Leetcode](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
    
9. Zigzag Traversal (medium) [Leetcode](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
    
10. Level Averages in a Binary Tree (easy) [Leetcode](https://leetcode.com/problems/average-of-levels-in-binary-tree/)
    
11. Minimum Depth of a Binary Tree (easy) [Leetcode](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
    
12. Maximum Depth of a Binary Tree (easy) [Leetcode](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
    
13. Level Order Successor (easy) [Geeksforgeeks](https://www.geeksforgeeks.org/level-order-successor-of-a-node-in-binary-tree/)
    
14. Connect Level Order Siblings (medium) [Leetcode](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
    
15. Problem Challenge 1: Connect All Level Order Siblings (medium) [Educative](https://www.educative.io/m/connect-all-siblings)
    
16. Problem Challenge 2: Right View of a Binary Tree (easy) [Leetcode](https://leetcode.com/problems/binary-tree-right-side-view/)
    

## **WEEK 6: Tree : DFS and Graph**

1. Introduction
    
2. Binary Tree Path Sum (easy) [Leetcode](https://leetcode.com/problems/path-sum/)
    
3. All Paths for a Sum (medium) [Leetcode](https://leetcode.com/problems/path-sum-iii/)
    
4. Sum of Path Numbers (medium) [Leetcode](https://leetcode.com/problems/sum-root-to-leaf-numbers/)
    
5. Path With Given Sequence (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/check-root-leaf-path-given-sequence/)
    
6. Count Paths for a Sum (medium) [Leetcode](https://leetcode.com/problems/path-sum-iii/)
    
7. Problem Challenge 1: Tree Diameter (medium) [Leetcode](https://leetcode.com/problems/diameter-of-binary-tree/)
    
8. Problem Challenge 2: Path with Maximum Sum (hard) [Leetcode](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
    
9. Introduction to Graph (Representations, Abstract Data Type (ADT))
    
10. Graph Traversal: Depth First Search(DFS)
    
11. Graph Traversal: Breadth First Search (BFS)
    
12. Find if Path Exists in Graph(easy) [Leetcode](https://leetcode.com/problems/find-if-path-exists-in-graph/)
    
13. Number of Provinces (medium) [Leetcode](https://leetcode.com/problems/number-of-provinces/)
    
14. Minimum Number of Vertices to Reach All Nodes(medium) [Leetcode](https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/)
    

## **WEEK 7: Island and Two Heaps**

1. Introduction to Island Pattern
    
2. Number of Islands (easy) [Leetcode](https://leetcode.com/problems/number-of-islands/)
    
3. Biggest Island (easy)
    
4. Flood Fill (easy) [Leetcode](https://leetcode.com/problems/flood-fill/)
    
5. Number of Closed Islands (easy) [Leetcode](https://leetcode.com/problems/number-of-closed-islands/)
    
6. Find the Median of a Number Stream (medium) [Leetcode](https://leetcode.com/problems/find-median-from-data-stream/)
    
7. Sliding Window Median (hard) [Leetcode](https://leetcode.com/problems/sliding-window-median/)
    
8. Maximize Capital (hard) [Leetcode](https://leetcode.com/problems/ipo/)
    
9. *_Maximum Sum Combinations_ (medium) [InterviewBit](https://www.interviewbit.com/problems/maximum-sum-combinations/)
    

## **WEEK 8: Subsets and Modified Binary Search**

1. Introduction [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/R87WmWYrELz)
    
2. Subsets (easy) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/gx2OqlvEnWG)
    
3. Subsets With Duplicates (easy) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/7npk3V3JQNr)
    
4. Permutations (medium) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/B8R83jyN3KY)
    
5. Introduction [Complete Pattern Theory and Solutions](https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions/blob/master/binary-search/BinarySearch.md)
    
6. Order-agnostic Binary Search (easy) [Geeksforgeeks](https://www.geeksforgeeks.org/order-agnostic-binary-search/)
    
7. Ceiling of a Number (medium) [Geeksforgeeks-Ceil](https://www.geeksforgeeks.org/ceiling-in-a-sorted-array/) [Geeksforgeeks-Floor](https://www.geeksforgeeks.org/floor-in-a-sorted-array/)
    
8. Next Letter (medium) [Leetcode](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)
    
9. Number Range (medium) [Leetcode](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
    
10. Search in a Sorted Infinite Array (medium) [Leetcode](https://www.geeksforgeeks.org/find-position-element-sorted-array-infinite-numbers/)
    
11. Minimum Difference Element (medium): Find the floor & ceil take the difference, minimum would be the ans
    
12. Bitonic Array Maximum (easy) [Geeksforgeeks](https://www.geeksforgeeks.org/find-the-maximum-element-in-an-array-which-is-first-increasing-and-then-decreasing/)
    
13. Problem Challenge 1: Search Bitonic Array (medium) [Leetcode](https://leetcode.com/problems/find-in-mountain-array/)
    
14. Problem Challenge 2: Search in Rotated Array (medium) [Leetcode](https://leetcode.com/problems/search-in-rotated-sorted-array/)
    
15. Problem Challenge 3: Rotation Count (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/find-rotation-count-rotated-sorted-array/)
    
16. *Search a 2D Matrix (medium) [Leetcode](https://leetcode.com/problems/search-a-2d-matrix/)
    
17. *Minimum Number of Days to Make m Bouquets (medium) [Leetcode](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/)
    
18. *Koko Eating Bananas (medium) [Leetcode](https://leetcode.com/problems/koko-eating-bananas/)
    
19. *Capacity To Ship Packages Within D Days (medium) [Leetcode](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)
    
20. *Median of Two Sorted Arrays (hard) [Leetcode](https://leetcode.com/problems/median-of-two-sorted-arrays/)
    

## **WEEK 9: Bitwise XOR and Top K Elements**

1. Single Number (easy)
    
2. Two Single Numbers (medium)
    
3. Complement of Base 10 Number (medium)
    
4. Problem Challenge 1: Flip and Invert an Image (hard)
    
5. [Introduction](https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions/blob/master/13.-pattern-top-k-elements/01.Introduction.md)
    
6. Top 'K' Numbers (easy) [Solution](https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions/blob/master/13.-pattern-top-k-elements/02.top-k-numbers.md)
    
7. Kth Smallest Number (easy)
    
8. 'K' Closest Points to the Origin (easy) [Leetcode](https://leetcode.com/problems/k-closest-points-to-origin/)
    
9. Connect Ropes (easy)
    
10. Top 'K' Frequent Numbers (medium)
    
11. Frequency Sort (medium)
    
12. Kth Largest Number in a Stream (medium) [Leetcode](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
    

## **WEEK 10: K-way merge and Greedy Sort**

1. Merge K Sorted Lists (medium) [Leetcode](https://leetcode.com/problems/merge-k-sorted-lists/)
    
2. Kth Smallest Number in M Sorted Lists (Medium) [Geeksforgeeks](https://www.geeksforgeeks.org/find-m-th-smallest-value-in-k-sorted-arrays/)
    
3. Kth Smallest Number in a Sorted Matrix (Hard) [Educative.io](https://www.educative.io/courses/grokking-the-coding-interview/x1NJVYKNvqz)
    
4. Smallest Number Range (Hard) [Leetcode](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/)
    
5. Valid Palindrome II (easy) [Leetcode](https://leetcode.com/problems/valid-palindrome-ii/)
    
6. Maximum Length of Pair Chain (medium) [Leetcode](https://leetcode.com/problems/maximum-length-of-pair-chain/)
    
7. Minimum Add to Make Parentheses Valid (medium) [Leetcode](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/)
    
8. Remove Duplicate Letters (medium) [Leetcode](https://leetcode.com/problems/remove-duplicate-letters/)
    
9. Largest Palindromic Number (Medium) [Leetcode](https://leetcode.com/problems/largest-palindromic-number/)
    
10. Removing Minimum and Maximum From Array (medium) [Leetcode](https://leetcode.com/problems/removing-minimum-and-maximum-from-array/)
    

## **WEEK 11: 0/1 Knapsack and BackTracking**

1. 0/1 Knapsack (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/)
    
2. Equal Subset Sum Partition (medium) [Leetcode](https://leetcode.com/problems/partition-equal-subset-sum/)
    
3. Subset Sum (medium) [Geeksforgeeks](https://www.geeksforgeeks.org/subset-sum-problem-dp-25/)
    
4. Minimum Subset Sum Difference (hard) [Geeksforgeeks](https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/)
    
5. Combination Sum (medium) [Leetcode - I](https://leetcode.com/problems/combination-sum/) [Leetcode - II](https://leetcode.com/problems/combination-sum-ii/) [Leetcode - III](https://leetcode.com/problems/combination-sum-iii/) [Leetcode - IV](https://leetcode.com/problems/combination-sum-iv/)
    
6. Word Search (medium) [Leetcode - I](https://leetcode.com/problems/word-search/) [Leetcode - II (Hard)](https://leetcode.com/problems/word-search-ii/)
    
7. Sudoku Solver (hard) [Leetcode](https://leetcode.com/problems/sudoku-solver/)
    
8. Factor Combinations (medium) [Leetcode+](https://leetcode.com/problems/factor-combinations/)
    
9. Split a String Into the Max Number of Unique Substrings (medium) [Leetcode](https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings/)
    

## **WEEK 12: Trie and Topological Sort**

1. Implement Trie (Prefix Tree) (medium) [Leetcode](https://leetcode.com/problems/implement-trie-prefix-tree/)
    
2. Index Pairs of a String (easy) [Leetcode+](https://leetcode.com/problems/index-pairs-of-a-string/)
    
3. Design Add and Search Words Data Structure (medium) [Leetcode](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
    
4. Extra Characters in a String (medium) [Leetcode](https://leetcode.com/problems/extra-characters-in-a-string/)
    
5. Search Suggestions System (medium) [Leetcode](https://leetcode.com/problems/search-suggestions-system/)
    
6. Topological Sort (medium) [Youtube](https://www.youtube.com/watch?v=cIBFEhD77b4)
    
7. Tasks Scheduling (medium) [Leetcode-Similar](https://leetcode.com/problems/course-schedule/)
    
8. Tasks Scheduling Order (medium) [Leetcode-Similar](https://leetcode.com/problems/course-schedule/)
    
9. All Tasks Scheduling Orders (hard) [Leetcode-Similar](https://leetcode.com/problems/course-schedule-ii/)
    
10. Alien Dictionary (hard) [Leetcode](https://leetcode.com/problems/alien-dictionary/)
    
11. Problem Challenge 1: Reconstructing a Sequence (hard) [Leetcode](https://leetcode.com/problems/sequence-reconstruction/)
    
12. Problem Challenge 2: Minimum Height Trees (hard) [Leetcode](https://leetcode.com/problems/minimum-height-trees/)
    

## **WEEK 13: Union Find , Ordered Set and Multi-thread**

1. Redundant Connection (medium) [Leetcode - I](https://leetcode.com/problems/redundant-connection/) [Leetcode - II (Hard)](https://leetcode.com/problems/redundant-connection-ii/)
    
2. Number of Provinces (medium) [Leetcode](https://leetcode.com/problems/number-of-provinces/)
    
3. Is Graph Bipartite? (medium) [Leetcode](https://leetcode.com/problems/is-graph-bipartite/)
    
4. Path With Minimum Effort (medium) [Leetcode](https://leetcode.com/problems/path-with-minimum-effort/)
    
5. Merge Similar Items (easy) [Leetcode](https://leetcode.com/problems/merge-similar-items/)
    
6. 132 Pattern (medium) [Leetcode](https://leetcode.com/problems/132-pattern/)
    
7. My Calendar I (medium) [Leetcode](https://leetcode.com/problems/my-calendar-i/) [Leetcode - II](https://leetcode.com/problems/my-calendar-ii/) [Leetcode - III (Hard)](https://leetcode.com/problems/my-calendar-iii/)
    

