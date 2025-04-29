import os
import ast

EXECUTOR_ROOT = os.getenv("EXECUTOR_ROOT","/datacanvas/titan_cv_agent_executor")

ACCESS_KEY = os.getenv("ACCESS_KEY", "")
PLANNER_LLM_MODEL_NAME = os.getenv("PLANNER_LLM_MODEL_NAME","Qwen/Qwen2.5-72B-Instruct")
PLANNER_LLM_URL = os.getenv("PLANNER_LLM_URL", "https://api.siliconflow.cn/v1/chat/completions")
CALL_FUNCTION_URL = os.getenv("CALL_FUNCTION_URL", "http://127.0.0.1:52001/call_function")

TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS",900))
EXECUTOR_FLASK_ADDR = os.getenv("EXECUTOR_FLASK_ADDR","0.0.0.0")
EXECUTOR_FLASK_PORT = int(os.getenv("EXECUTOR_FLASK_PORT",52002))
EXECUTOR_OUTPUT_DIR = os.getenv("EXECUTOR_OUTPUT_DIR",os.path.join(EXECUTOR_ROOT, "output/"))

model_name = PLANNER_LLM_MODEL_NAME.split("/")[-1]
RESULT_PREFIX = os.getenv("RESULT_PREFIX", os.path.join(EXECUTOR_ROOT,f"runtime/running_result"))
LOG_PATH_PREFIX = os.getenv("LOG_PATH_PREFIX", os.path.join(EXECUTOR_ROOT,f"runtime/logs/"))
FUNCTION_POOL_DIR = os.getenv("FUNCTION_POOL_DIR", os.path.join(EXECUTOR_ROOT,f"runtime/function_pool/"))
IMAGE_SEARCH_URL = os.getenv("IMAGE_SEARCH_URL", "https://www.bing.com/images/search")
BASE_FUNCTION_SCHEMA_PATH = os.getenv("BASE_FUNCTION_SCHEMA_PATH", os.path.join(EXECUTOR_ROOT, "exist_func_def.json"))


BASE_FUNCTIONS = ["preprocess","postprocess","detection","segmentation","classification","counting","tracking","pose","optical_flow","ocr","vlm","llm","alarm","output","videoprecess"]
GET_PLAN_QUERY_PREFIX = os.getenv("GET_PLAN_QUERY_PREFIX", """你是一个计算机视觉专家，可以通过多步骤串联完成一个较为复杂的计算机视觉任务。【function schema】描述了我们有的各项已经具备的、计算机视觉基础的【base function】，以及它们的输入、输出参数情况。 现在请你根据【任务需求】【function schema】串联完成计算机视觉任务，你需要创造必要的【adapter function】以完成【base function】之间的类型、格式转换或实现我们尚未具备的功能，同时需要写出其具体的可运行的python代码，以确保各步骤可以顺利的串联执行。 完成任务时请遵守以下约定： 1. 如果使用【base function】请严格遵守【function schema】的输出，不要创造不存在的输出变量。 2. 不要任何解释和说明。 3. 如果有输出，所有文件请保存到os.getenv('AGENT_OUTPUT_DIR')。  4.请严格遵循用户输入输出格式，需要输入时使用<|input_step_x.xxx|>占位，需要使用之前step的输出则使用<|input_step_x.xxx|>占位(x是当前step，xxx是需要的变量名称)，不能使用其它格式。 5.query_list所有项必须是英文。6.【adapter function】记得导入需要的包，例如import os等。以下是一个例子： 【任务需求开始】 请制作一个人群密集分析agent，通过无人机拍摄的照片进行分析，判断照片中桥上的人群是否密集，需支持密集判定阈值输入。 【任务需求结束】 {\"pipeline\": [{\"step\": \"检测桥并获取检测结果。\", \"function\": \"detection\", \"input\": {\"image_path_list\": \"<|input_step_1.image_path_list|>\", \"query_list\": [\"bridge\"]}}, {\"step\": \"处理边界框信息，删除每个边界框最后的信心值和类别信息，返回仅包含边界框坐标的列表。\", \"function\": \"get_pure_bbox_list\", \"input\": {\"raw_bbox_data\": \"<|output_step_1.boxes_list|>\"}}, {\"step\": \"按照所有边界框给定的坐标进行裁切，仅保留需要识别的区域，返回切片后的图片路径列表。\", \"function\": \"batch_crop_images\", \"input\": {\"image_path_list\": \"<|output_step_1.image_path_list|>\", \"crop_bboxes\": \"<|output_step_2.pure_bbox_list|>\"}}, {\"step\": \"对区域中的人群进行数量统计。\", \"function\": \"counting\", \"input\": {\"image_path_list\": \"<|output_step_3.all_croped_image|>\", \"query_list\": [\"people\"]}}, {\"step\": \"判断照片中桥上的人群是否密集。\", \"function\": \"llm\", \"input\": {\"query\": [\"你是一个专业的环境状态报告员，需要告诉用户所处环境人员是否密集？（密集判断标准是:\", \"<|input_step_5.threshold|>\", \"）实际统计结果:\", \"<|output_step_4.counting_sum|>\", \"请你依据上述信息向用户报告。\"]}}], \"adapter_function\": {\"get_pure_bbox_list.py\": \"def get_pure_bbox_list(raw_bbox_data):\\n    all_bbox = []\\n    for single_image in raw_bbox_data:\\n        single_image_bbox = []\\n        for bbox in single_image:\\n            single_image_bbox.append(bbox[:4])\\n        all_bbox.append(single_image_bbox)\\n    return {\\\"pure_bbox_list\\\":all_bbox}\", \"batch_crop_images.py\": \"import os\\nimport cv2\\nimport uuid \\n\\ndef batch_crop_images(image_path_list, crop_bboxes):\\n    all_croped_image = []\\n    src = None\\n    for idx, single_image in enumerate(image_path_list):\\n        single_image_crop_bboxes = crop_bboxes[idx]\\n        if single_image_crop_bboxes is not None:\\n            for bbox in single_image_crop_bboxes:\\n                src = cv2.imread(single_image)\\n                src = src[int(float(bbox[1])):int(float(bbox[3])),int(float(bbox[0])):int(float(bbox[2]))]\\n                if src is not None:\\n                    croped_image_path = os.getenv('AGENT_OUTPUT_DIR') + str(uuid.uuid4()) + \\\".jpg\\\"\\n                    if src is not None:\\n                        cv2.imwrite(croped_image_path, src)\\n                        all_croped_image.append(croped_image_path)\\n    print(all_croped_image)\\n    return {\\\"all_croped_image\\\" : all_croped_image}\"}} 下面开始执行任务：""")

DEFAULT_IMAGE_LIST = ast.literal_eval(os.getenv("DEFAULT_IMAGE_LIST")) if os.getenv("DEFAULT_IMAGE_LIST") else ["/models/baohan/data/images/sample.jpg"]
DEFAULT_VIDEO_PATH = os.getenv("DEFAULT_VIDEO_PATH", "sample.mp4")
DEFAULT_RTMP_PATH = os.getenv("DEFAULT_RTMP_PATH", "sample") 
DEFAULT_BBOX = ast.literal_eval(os.getenv("DEFAULT_BBOX")) if os.getenv("DEFAULT_BBOX") else [0, 0, 0, 0]
DEFAULT_QUERY = os.getenv("DEFAULT_QUERY", "people")
DEFAULT_VALUE = os.getenv("DEFAULT_VALUE", "20")
DEFAULT_THRESHOLD = int(os.getenv("DEFAULT_THRESHOLD", 5))
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", 0.5))