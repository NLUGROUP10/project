import collections
import json


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


    @staticmethod
    def serialize(root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        queue = collections.deque([root])
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('None')
        return json.dumps(res)


    @staticmethod
    def deserialize(data:str):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        dataList = json.loads(data)
        root = Node(dataList[0])
        queue = collections.deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if dataList[i] != 'None':
                node.left = Node(dataList[i])
                queue.append(node.left)
            i += 1
            if dataList[i] != 'None':
                node.right = Node(dataList[i])
                queue.append(node.right)
            i += 1
        return root

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    @staticmethod
    def pre_order_traverse(root):
        res = []

        def helper(root, depth=None):
            if depth is None:
                depth = "/"
            if root != None:
                res.append((root.val, depth[:]))
                branchleft = depth[:]
                branchleft = branchleft+"0"
                helper(root.left, branchleft)
                branchright = depth[:]
                branchright = branchright+"1"
                helper(root.right, branchright)
        helper(root)
        return res

    @staticmethod
    def add_root_node(root):
        start = Node(json.dumps(["start_2","empty"]))
        end   = Node(json.dumps(["end_0","empty"]))
        start.right = end
        start.left = root
        return start
