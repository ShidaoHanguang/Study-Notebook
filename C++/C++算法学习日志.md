# C++算法学习日志

​	本日志为学习[代码随想录](https://www.programmercarl.com/)的记录，预期分20天时间学习所有涉及的数据结构和热门算法。本日志会在基于原内容的情况下，尽量添加一些自己的想法，以便回顾时使用。

## 数组

### 1.基础

​	数组，再简单不过，其为**存放在连续内存空间上的相同类型数据的集合。**可以通过**索引**的方式查询对应的数值。

#### 注意事项

- 数组下标从**0**开始
- 数组的存储空间是**连续**的



### 2.二分查找

#### 题目描述

给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。

示例 1:

```text
输入: nums = [-1,0,3,5,9,12], target = 9     
输出: 4       
解释: 9 出现在 nums 中并且下标为 4     
```

示例 2:

```text
输入: nums = [-1,0,3,5,9,12], target = 2     
输出: -1        
解释: 2 不存在 nums 中因此返回 -1        
```

提示：

- 你可以假设 nums 中的所有元素是不重复的。
- n 将在 [1, 10000]之间。
- nums 的每个元素都将在 [-9999, 9999]之间。



#### 思路分析

**有序无重复数组**是关键，这意味着数组的索引顺序本身意味着大小，因此可以采用**二分法**的方式进行查找。通过建立左右索引，判断其中间值与所搜寻值的比较，移动左右索引的区间，进而逐渐锁定所搜寻值的位置，最终通过迭代获取结果。



#### 注意事项

二分法查找时需要对定义域区间多加注意，谨防对区间误判导致的对**中间值的遗漏**。



#### 代码展示

```cpp
// 版本二
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size(); // 定义target在左闭右开的区间里，即：[left, right)
        while (left < right) { // 因为left == right的时候，在[left, right)是无效的空间，所以使用 <
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle; // target 在左区间，在[left, middle)中
            } else if (nums[middle] < target) {
                left = middle + 1; // target 在右区间，在[middle + 1, right)中
            } else { // nums[middle] == target
                return middle; // 数组中找到目标值，直接返回下标
            }
        }
        // 未找到目标值
        return -1;
    }
};
```



### 3. 元素移除

#### 题目描述

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并**原地**修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

#### 思路分析

双指针法（快慢指针法）： **通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。**

定义快慢指针

- 快指针：寻找新数组的元素 ，新数组就是不含有目标元素的数组
- 慢指针：指向更新 新数组下标的位置

#### 代码展示

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slowIndex = 0;
        for (int fastIndex = 0; fastIndex < nums.size(); fastIndex++) {
            if (val != nums[fastIndex]) {
                nums[slowIndex++] = nums[fastIndex];
            }
        }
        return slowIndex;
    }
};
```

