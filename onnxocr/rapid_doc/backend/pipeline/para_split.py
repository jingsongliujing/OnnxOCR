import copy
from onnxocr.rapid_doc.utils.enum_class import ContentType, BlockType, SplitFlag
from onnxocr.rapid_doc.utils.language import detect_lang


LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；')
LIST_END_FLAG = ('.', '。', ';', '；')


class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'

def __process_blocks(blocks):
    """
    对所有 block 进行预处理:
    1. 对 text 类型根据 lines 重算 bbox
    2. 连续 text block 合并成一个 group
    3. 非 text block (image, title, interline_equation 等) 单独作为 group
    4. 每个 group 都有 group_type 标记，避免下游误判
    """

    result = []
    current_group = []

    def flush_current_group():
        """把 current_group 收集到 result"""
        nonlocal current_group
        if current_group:
            result.append({
                "group_type": "text",
                "blocks": current_group
            })
            current_group = []

    for i, current_block in enumerate(blocks):

        if current_block["type"] == "text":
            # 重算 bbox_fs
            current_block["bbox_fs"] = copy.deepcopy(current_block["bbox"])
            if "lines" in current_block and len(current_block["lines"]) > 0:
                current_block["bbox_fs"] = [
                    min(line["bbox"][0] for line in current_block["lines"]),
                    min(line["bbox"][1] for line in current_block["lines"]),
                    max(line["bbox"][2] for line in current_block["lines"]),
                    max(line["bbox"][3] for line in current_block["lines"]),
                ]
            # 累积到当前 text group
            current_group.append(current_block)

        else:
            # 遇到非 text block，先把前面的 text group flush 掉
            flush_current_group()

            # 当前 block 单独作为 group
            result.append({
                "group_type": current_block["type"],  # image / title / interline_equation 等
                "blocks": [current_block]
            })

        # 如果下一个是 title / interline_equation，要切断 text group
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            if next_block["type"] in ["title", "interline_equation"]:
                flush_current_group()

    # 收尾，处理残余的 text group
    flush_current_group()

    return result


def __is_list_or_index_block(block):
    # 一个block如果是list block 应该同时满足以下特征
    # 1.block内有多个line 2.block 内有多个line左侧顶格写 3.block内有多个line 右侧不顶格（狗牙状）
    # 1.block内有多个line 2.block 内有多个line左侧顶格写 3.多个line以endflag结尾
    # 1.block内有多个line 2.block 内有多个line左侧顶格写 3.block内有多个line 左侧不顶格

    # index block 是一种特殊的list block
    # 一个block如果是index block 应该同时满足以下特征
    # 1.block内有多个line 2.block 内有多个line两侧均顶格写 3.line的开头或者结尾均为数字
    if len(block['lines']) >= 2:
        first_line = block['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block_weight = block['bbox_fs'][2] - block['bbox_fs'][0]
        block_height = block['bbox_fs'][3] - block['bbox_fs'][1]
        page_weight, page_height = block['page_size']

        left_close_num = 0
        left_not_close_num = 0
        right_not_close_num = 0
        right_close_num = 0
        lines_text_list = []
        center_close_num = 0
        external_sides_not_close_num = 0
        multiple_para_flag = False
        last_line = block['lines'][-1]

        if page_weight == 0:
            block_weight_radio = 0
        else:
            block_weight_radio = block_weight / page_weight
        # logger.info(f"block_weight_radio: {block_weight_radio}")

        # 如果首行左边不顶格而右边顶格,末行左边顶格而右边不顶格 （第一行可能可以右边不顶格）
        if (
            first_line['bbox'][0] - block['bbox_fs'][0] > line_height / 2
            and abs(last_line['bbox'][0] - block['bbox_fs'][0]) < line_height / 2
            and block['bbox_fs'][2] - last_line['bbox'][2] > line_height
        ):
            multiple_para_flag = True

        block_text = ''

        for line in block['lines']:
            line_text = ''

            for span in line['spans']:
                span_type = span['type']
                if span_type == ContentType.TEXT:
                    line_text += span['content'].strip()
            # 添加所有文本，包括空行，保持与block['lines']长度一致
            lines_text_list.append(line_text)
            block_text = ''.join(lines_text_list)

        block_lang = detect_lang(block_text)
        # logger.info(f"block_lang: {block_lang}")

        for line in block['lines']:
            line_mid_x = (line['bbox'][0] + line['bbox'][2]) / 2
            block_mid_x = (block['bbox_fs'][0] + block['bbox_fs'][2]) / 2
            if (
                line['bbox'][0] - block['bbox_fs'][0] > 0.7 * line_height
                and block['bbox_fs'][2] - line['bbox'][2] > 0.7 * line_height
            ):
                external_sides_not_close_num += 1
            if abs(line_mid_x - block_mid_x) < line_height / 2:
                center_close_num += 1

            # 计算line左侧顶格数量是否大于2，是否顶格用abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height/2 来判断
            if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                left_close_num += 1
            elif line['bbox'][0] - block['bbox_fs'][0] > line_height:
                left_not_close_num += 1

            # 计算右侧是否顶格
            if abs(block['bbox_fs'][2] - line['bbox'][2]) < line_height:
                right_close_num += 1
            else:
                # 类中文没有超长单词的情况，可以用统一的阈值
                if block_lang in ['zh', 'ja', 'ko']:
                    closed_area = 0.26 * block_weight
                else:
                    # 右侧不顶格情况下是否有一段距离，拍脑袋用0.3block宽度做阈值
                    # block宽的阈值可以小些，block窄的阈值要大
                    if block_weight_radio >= 0.5:
                        closed_area = 0.26 * block_weight
                    else:
                        closed_area = 0.36 * block_weight
                if block['bbox_fs'][2] - line['bbox'][2] > closed_area:
                    right_not_close_num += 1

        # 判断lines_text_list中的元素是否有超过80%都以LIST_END_FLAG结尾
        line_end_flag = False
        # 判断lines_text_list中的元素是否有超过80%都以数字开头或都以数字结尾
        line_num_flag = False
        num_start_count = 0
        num_end_count = 0
        flag_end_count = 0

        if len(lines_text_list) > 0:
            for line_text in lines_text_list:
                if len(line_text) > 0:
                    if line_text[-1] in LIST_END_FLAG:
                        flag_end_count += 1
                    if line_text[0].isdigit():
                        num_start_count += 1
                    if line_text[-1].isdigit():
                        num_end_count += 1

            if (
                num_start_count / len(lines_text_list) >= 0.8
                or num_end_count / len(lines_text_list) >= 0.8
            ):
                line_num_flag = True
            if flag_end_count / len(lines_text_list) >= 0.8:
                line_end_flag = True

        # 有的目录右侧不贴边, 目前认为左边或者右边有一边全贴边，且符合数字规则极为index
        if (
            left_close_num / len(block['lines']) >= 0.8
            or right_close_num / len(block['lines']) >= 0.8
        ) and line_num_flag:
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.INDEX

        # 全部line都居中的特殊list识别，每行都需要换行，特征是多行，且大多数行都前后not_close,每line中点x坐标接近
        # 补充条件block的长宽比有要求
        elif (
            external_sides_not_close_num >= 2
            and center_close_num == len(block['lines'])
            and external_sides_not_close_num / len(block['lines']) >= 0.5
            and block_height / block_weight > 0.4
        ):
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.LIST

        elif (
            left_close_num >= 2
            and (right_not_close_num >= 2 or line_end_flag or left_not_close_num >= 2)
            and not multiple_para_flag
            # and block_weight_radio > 0.27
        ):
            # 处理一种特殊的没有缩进的list，所有行都贴左边，通过右边的空隙判断是否是item尾
            if left_close_num / len(block['lines']) > 0.8:
                # 这种是每个item只有一行，且左边都贴边的短item list
                if flag_end_count == 0 and right_close_num / len(block['lines']) < 0.5:
                    for line in block['lines']:
                        if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                # 这种是大部分line item 都有结束标识符的情况，按结束标识符区分不同item
                elif line_end_flag:
                    for i, line in enumerate(block['lines']):
                        if (
                            len(lines_text_list[i]) > 0
                            and lines_text_list[i][-1] in LIST_END_FLAG
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            if i + 1 < len(block['lines']):
                                block['lines'][i + 1][
                                    ListLineTag.IS_LIST_START_LINE
                                ] = True
                # line item基本没有结束标识符，而且也没有缩进，按右侧空隙判断哪些是item end
                else:
                    line_start_flag = False
                    for i, line in enumerate(block['lines']):
                        if line_start_flag:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                            line_start_flag = False

                        if (
                            abs(block['bbox_fs'][2] - line['bbox'][2])
                            > 0.1 * block_weight
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            line_start_flag = True
            # 一种有缩进的特殊有序list,start line 左侧不贴边且以数字开头，end line 以 IS_LIST_END_FLAG 结尾且数量和start line 一致
            elif num_start_count >= 2 and num_start_count == flag_end_count:
                for i, line in enumerate(block['lines']):
                    if len(lines_text_list[i]) > 0:
                        if lines_text_list[i][0].isdigit():
                            line[ListLineTag.IS_LIST_START_LINE] = True
                        if lines_text_list[i][-1] in LIST_END_FLAG:
                            line[ListLineTag.IS_LIST_END_LINE] = True
            else:
                # 正常有缩进的list处理
                for line in block['lines']:
                    if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                        line[ListLineTag.IS_LIST_START_LINE] = True
                    if abs(block['bbox_fs'][2] - line['bbox'][2]) > line_height:
                        line[ListLineTag.IS_LIST_END_LINE] = True

            return BlockType.LIST
        else:
            return BlockType.TEXT
    else:
        return BlockType.TEXT


def __merge_2_text_blocks(block1, block2):
    if len(block1['lines']) > 0:
        first_line = block1['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block1_weight = block1['bbox'][2] - block1['bbox'][0]
        block2_weight = block2['bbox'][2] - block2['bbox'][0]
        min_block_weight = min(block1_weight, block2_weight)
        if abs(block1['bbox_fs'][0] - first_line['bbox'][0]) < line_height / 2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_height = last_line['bbox'][3] - last_line['bbox'][1]
                if len(first_line['spans']) > 0:
                    first_span = first_line['spans'][0]
                    if len(first_span['content']) > 0:
                        span_start_with_num = first_span['content'][0].isdigit()
                        span_start_with_big_char = first_span['content'][0].isupper()
                        if (
                            # 上一个block的最后一个line的右边界和block的右边界差距不超过line_height
                            abs(block2['bbox_fs'][2] - last_line['bbox'][2]) < line_height
                            # 上一个block的最后一个span不是以特定符号结尾
                            and not last_span['content'].endswith(LINE_STOP_FLAG)
                            # 两个block宽度差距超过2倍也不合并
                            and abs(block1_weight - block2_weight) < min_block_weight
                            # 下一个block的第一个字符是数字
                            and not span_start_with_num
                            # 下一个block的第一个字符是大写字母
                            and not span_start_with_big_char
                        ):
                            if block1['page_num'] != block2['page_num']:
                                for line in block1['lines']:
                                    for span in line['spans']:
                                        span[SplitFlag.CROSS_PAGE] = True
                            block2['lines'].extend(block1['lines'])
                            block1['lines'] = []
                            block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __merge_2_list_blocks(block1, block2):
    if block1['page_num'] != block2['page_num']:
        for line in block1['lines']:
            for span in line['spans']:
                span[SplitFlag.CROSS_PAGE] = True
    block2['lines'].extend(block1['lines'])
    block1['lines'] = []
    block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __is_list_group(text_blocks_group):
    # list group的特征是一个group内的所有block都满足以下条件
    # 1.每个block都不超过3行 2. 每个block 的左边界都比较接近(逻辑简单点先不加这个规则)
    for block in text_blocks_group:
        if len(block['lines']) > 3:
            return False
    return True

def __para_merge_page(blocks):
    page_text_blocks_groups = __process_blocks(blocks)

    for group in page_text_blocks_groups:
        blocks_in_group = group["blocks"]

        if len(blocks_in_group) > 0 and group["group_type"] == "text":
            # 只对 text group 判断是否为 list/index
            for block in blocks_in_group:
                block_type = __is_list_or_index_block(block)
                block["type"] = block_type

        if len(blocks_in_group) > 1 and group["group_type"] == "text":
            # 在合并前判断这个 group 是否是一个 list group
            is_list_group = __is_list_group(blocks_in_group)

            # 倒序遍历
            for i in range(len(blocks_in_group) - 1, -1, -1):
                current_block = blocks_in_group[i]

                if i - 1 >= 0:
                    prev_block = blocks_in_group[i - 1]

                    if (
                        current_block["type"] == "text"
                        and prev_block["type"] == "text"
                        and not is_list_group
                    ):
                        __merge_2_text_blocks(current_block, prev_block)

                    elif (
                        (current_block["type"] == BlockType.LIST and prev_block["type"] == BlockType.LIST)
                        or (current_block["type"] == BlockType.INDEX and prev_block["type"] == BlockType.INDEX)
                    ):
                        __merge_2_list_blocks(current_block, prev_block)


def para_split(page_info_list):
    all_blocks = []
    for page_info in page_info_list:
        blocks = copy.deepcopy(page_info['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_info['page_idx']
            block['page_size'] = page_info['page_size']
        all_blocks.extend(blocks)

    __para_merge_page(all_blocks)
    for page_info in page_info_list:
        page_info['para_blocks'] = []
        for block in all_blocks:
            if 'page_num' in block:
                if block['page_num'] == page_info['page_idx']:
                    page_info['para_blocks'].append(block)
                    # 从block中删除不需要的page_num和page_size字段
                    del block['page_num']
                    del block['page_size']


if __name__ == '__main__':
    input_blocks = []
    # 调用函数
    groups = __process_blocks(input_blocks)
    for group_index, group in enumerate(groups):
        print(f'Group {group_index}: {group}')
