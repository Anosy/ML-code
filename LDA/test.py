class Solution:
    """
    @param: : An integer
    @param: : An integer
    @return: An integer denote the count of digit k in 1..n
    """

    def digitCounts(self, k, n):
        if k == 0 and n == 0:  # 计算特殊情况0
            return 1
        if k==0 and n//10==0:
            return 1
        out = 0
        result = 0
        num_list0 = []
        num_list1 = []
        num_list2 = []
        length = len(str(n))
        for i in range(0, length):
            div_num = 10 ** i
            a = n // div_num % 10  # 计算各个位的值，常用！
            b = n // (div_num * 10)  # 计算当前位置的高位
            c = n % (div_num * 10)  # 计算当前位置的后位
            num_list0.append(a)
            num_list1.append(b)
            num_list2.append(c)
        num_list2.insert(0, 0)  # 当位个位数的情况下，需要考虑将其的后位设置位0

        for each in range(length):
            wei = 10 ** each
            # 分类讨论
            if num_list0[each] > k:
                result = (num_list1[each] + 1) * wei
            elif num_list0[each] < k:
                result = num_list1[each] * wei
            else:
                result = num_list1[each] * wei + num_list2[each] + 1
            out += result
        if k == 0:
            # 排除如果出现k=0的情况下，可能计算出09，08，07也算上，即减去当前的高位
            out = out - result
        return out

    def digitCounts1(self, k, n):
        temp = n
        cnt = 0
        power = 1
        if k==0 and n==0:
            return 1
        if k==0 and n//10==0:
            return 1
        while temp!=0:
            digit = temp %  10 # digit当前位置值, temp
            if digit < k :
                cnt += (temp // 10 ) *power
            elif digit == k:
                cnt += (temp // 10) *power + 1 + (n - temp * power)
            else:
                if not (k==0 and temp//10==0):
                    cnt += (temp // 10 + 1) * power
            temp //= 10
            power *= 10
        return cnt


if __name__ == '__main__':
    s = Solution()
    k = s.digitCounts(0, 7)
    print(k)