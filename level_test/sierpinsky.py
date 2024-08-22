# --------------------------------------------
# 5) 다음 패턴을 찍는 함수 sierpinski_triangle을 짜 보세요. 
# 
# n = 2
#         *
#        * *
#       *   *
#      * * * *
#     *       * 
#    * *     * *
#   *   *   *   * 
#  * * * * * * * * 
# 
# n = 3 
#                 *
#                * *
#               *   *
#              * * * *
#             *       * 
#            * *     * *
#           *   *   *   * 
#          * * * * * * * *
#         *               *   
#        * *             * *  
#       *   *           *   * 
#      * * * *         * * * *
#     *       *       *       *  
#    * *     * *     * *     * *
#   *   *   *   *   *   *   *   * 
#  * * * * * * * * * * * * * * * *
# --------------------------------------------

# write your code here 
def sierpinski_triangle(n):
    return '\n'.join(s(n))

def triangle():
    res = [
      '   *    ',
      '  * *   ',
      ' *   *  ',
      '* * * * '
    ]

    return res

def s(n):
    if n == 1:
        return triangle()
    
    t = s(n-1)
    l = len(t[0])
    margin = ' ' * (1//2)

    res = []

    for line in t:
        res.append(margin + line + margin)
    for line in t:
        res.append(line + line)

    return res

for i in range(4):
    print(sierpinski_triangle(i+1))