import os
import re
import cv2
import json
import uuid
import time
import requests
import traceback
import threading
from bs4 import BeautifulSoup
from typing import List,Dict,Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ap_config import *
from ap_error_message import *

import logging
from datetime import datetime
now = datetime.now()
time_rec = now.strftime('%Y%m%d%H%M%S')
os.makedirs(LOG_PATH_PREFIX, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'{LOG_PATH_PREFIX}_{time_rec}.log'),
        logging.StreamHandler()
    ]
)

class AutoPipeline:
    def __init__(self):
            self.global_result_dict = {}
            self.global_response_str = ""
            self.real_image_list = None
            self.real_video_path = ""

    def timeout(self, seconds:int = TIMEOUT_SECONDS):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = [None]
                exception = [None]
                def runner():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                thread = threading.Thread(target=runner)
                thread.start()
                thread.join(seconds)
                if thread.is_alive():
                    raise TimeoutError(f"Function '{func.__name__}' exceeded {seconds} seconds.")
                if exception[0]:
                    raise exception[0]
                return result[0]
            return wrapper
        return decorator

    def get_response(self, url:str = None, data:Dict = None, is_call_fn=False):
        if url is None or data is None:
            raise ValueError(REQUEST_NONE_ERROR)
        try:
            if ACCESS_KEY is not None and ACCESS_KEY!="" and is_call_fn is False:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "Accept-Charset": "utf-8",
                    "Authorization": f"Bearer {ACCESS_KEY}"
                }
            else:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "Accept-Charset": "utf-8"
                }
            logging.info("======Sand Details======")
            logging.info(headers)
            logging.info(url)
            logging.info(data)
            logging.info("======End Details======")
            response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)
            logging.info("response.text>>>>>>>>>>>>>>>>>")
            logging.info(response.text)
            return json.loads(response.text)
        except Exception as e:
            logging.error(f"Data sent error:{e}")

    def call_function(self, idx:int = 1, step:Dict = None):
        if step is None:
            raise ValueError(CALL_FUNCTION_NONE_ERROR)
        step_result = self.get_response(CALL_FUNCTION_URL, step, True)
        step_code = step_result["code"]
        step_data = step_result["data"]
        
        if step_code == 1 and step_data is not None:
            step_data = {f'output_step_{idx}.{k}': v for k, v in step_data.items()}
        elif step_code == 1 and step_data is None:
            step_code = -1
            step_data = {f'output_step_{idx}.err_message':NONE_ERR_INFO}
        else:
            step_code = -1
            step_data = {f'output_step_{idx}.err_message':step_data}

        return step_code, step_data
        
    def _load_function_schema(self, file_path:str = None):
        with open(file_path, 'r', encoding='utf-8') as file:
            schema = json.load(file)
        function_schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        return function_schema_str

    def get_response_from_llm_based_on_schema(self, query:str = None, error_history = []):
        function_schema = self._load_function_schema(BASE_FUNCTION_SCHEMA_PATH)
        function_schema_str = f"Here is a function schema:\n{function_schema}\n\n"
        system_query = function_schema_str + GET_PLAN_QUERY_PREFIX
        
        messages = [
                {"role": "system", "content": system_query},
                {"role": "user", "content": query}
            ]
        
        if error_history != []:
            messages += error_history

        request = {
            "model": PLANNER_LLM_MODEL_NAME, 
            "messages":messages,
            "temperature": 0,
            "stream": False
        }

        response = self.get_response(url = PLANNER_LLM_URL, data = request, is_call_fn=False)
        print("==========================")
        print(type(response))
        print(response)
        return response['choices'][0]['message']['content'], error_history

    def get_real_media(self, pipeline:dict = None):
        #pipeline = [{'step': '对视频进行预处理，抽取视频中的帧图像。', 'function': 'videoprecess', 'input': {'method': 'get_frames', 'video_path': '<|input_step_1.video_path|>', 'interval': 1}}, {'step': '对裸土覆盖区域进行裸土和绿膜检测。', 'function': 'detection', 'input': {'image_path_list': '<|output_step_1.processed_path_list|>', 'query_list': ['naked_soil', 'green_membrane']}}, {'step': '通过检测结果判断裸土覆盖状态，确定是否触发报警。', 'function': 'alarm', 'input': {'result_obj': '<|output_step_2.boxes_list|>', 'detect_class': ['naked_soil'], 'detect_method': 'disappear', 'notice_email_or_tel': '<|input_step_3.notice_email_or_tel|>', 'alarm_threshold': 1}}, {'step': '输出报警结果。', 'function': 'output', 'input': {'result_obj': '<|output_step_3.is_alarm_success|>', 'method': 'show'}}]
        all_query_list = []
        for step in pipeline:
            if "query_list" in step["input"]:
                if (isinstance(step["input"]["query_list"],str)) and (not step["input"]["query_list"].startswith("<")):
                    all_query_list += step["input"]["query_list"]
                elif (isinstance(step["input"]["query_list"],List)):
                    all_query_list += step["input"]["query_list"]
                else:
                    logging.info(f"#4_1_0: Undefined query_list type.")

        try:
            if all_query_list != []:
                media = self._get_query_video(all_query_list)
                real_image_list = media["image_paths"]
                real_video_path = media["video_path"]
            else:
                real_image_list = DEFAULT_IMAGE_LIST
                real_video_path = DEFAULT_VIDEO_PATH
                logging.error(f"Not query media: {str(e)}")
        except Exception as e:
            real_image_list = DEFAULT_IMAGE_LIST
            real_video_path = DEFAULT_VIDEO_PATH
            logging.error(f"Change step input error: {str(e)}")
        return real_image_list, real_video_path

    def change_step_input(self, step):
        pattern = re.compile(r"<\|input_step_\d+\.[a-zA-Z_][\w]*\|>")
        
        def process_value(value):
            if isinstance(value, str) and pattern.match(value):
                if "=" in value:
                    return value.split("=")[-1]
                else:
                    if "image" in value:
                        return self.real_image_list
                        # if "query_list" in step["input"]:
                        #     real_images = _get_query_images(query_list=step["input"]["query_list"])
                    elif "video" in value:
                        return self.real_video_path
                    elif "rtmp_url" in value:
                        return DEFAULT_RTMP_PATH
                    elif "query" in value:
                        return DEFAULT_QUERY
                    elif "conf" in value or "confidence" in value:
                        return DEFAULT_CONF
                    elif "threshold" in value or "interval" in value or "duration" in value:
                        return DEFAULT_THRESHOLD
                    elif "bbox" in value:
                        return DEFAULT_BBOX
                    elif "resize_dims" in value:
                        return (256,256)
                    else:
                        return DEFAULT_VALUE
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        item = step["input"]
        for key, value in item.items():
            item[key] = process_value(value)
        
        return step

    def change_step_output(self, step):
        pattern = re.compile(r"<\|output_step_\d+\.[a-zA-Z_][\w]*\|>")
        
        def process_value(value):
            if isinstance(value, str) and pattern.match(value):
                varible_name = value.strip('<|').strip('|>')
                return self.global_result_dict.get(varible_name, value)
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        for key, value in step["input"].items():
            step["input"][key] = process_value(value)
        
        return step

    def filter_llm_output(self, llm_output: str = None):
        if llm_output is not None and "```" in llm_output:
            pattern = r'```(?:python|json)(.*?)```'
            matches = re.findall(pattern, llm_output, re.DOTALL)
            if matches:
                longest_item = max(matches, key=len)
                llm_output = longest_item
        llm_output = llm_output.replace("<AGENT_PIPELINE>","").replace("</AGENT_PIPELINE>","").replace("<\/AGENT_PIPELINE>","")
        return llm_output

    def write_jsonl(self, data:Dict = None, file_path:str = None):
        if data is not None and file_path is not None:
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')

    def _fetch_image_urls(self, search_query, num_images):
        search_url = f"{IMAGE_SEARCH_URL}?q={search_query}&count={num_images}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        image_urls = []
        for img in soup.find_all('img', class_='mimg'):
            img_url = img.get('src')
            if img_url and img_url.startswith('http'):
                image_urls.append(img_url)
        
        return image_urls

    def _download_images(self, image_urls, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saved_paths = []
        
        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                filename = os.path.join(save_dir, f'{str(uuid.uuid4())}.jpg')
                with open(filename, 'wb') as file:
                    file.write(response.content)
                    
                saved_paths.append(filename)
                print(f"Downloaded: {filename}")
            
            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
        return saved_paths

    def _get_query_images(self, query_list:Union[str, List[str]] = None, save_dir_prefix="search_image", num_images = 5):
        query_list = [query_list] if isinstance(query_list, str) else query_list
        search_image_list = []
        for query in query_list:
            save_dir = os.path.join(EXECUTOR_OUTPUT_DIR, save_dir_prefix, query)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
            image_urls = self._fetch_image_urls(query, num_images)
            paths = self._download_images(image_urls[:num_images], save_dir)
            search_image_list+=paths
        return search_image_list

    def _get_query_video(self, query_list:Union[str, List[str]] = None, resize_dims = (640, 360), save_dir_prefix:str="search_video"):
        logging.info(f"#4_1_1: Prepear video with query_list: {query_list}")
        from moviepy.editor import ImageSequenceClip
        image_files = self._get_query_images(query_list=query_list)
        for image in image_files:
            src = cv2.imread(image)
            src = cv2.resize(src, resize_dims, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(image, src)
        logging.info(f"#4_1_2: write images : {image_files}")
        clip = ImageSequenceClip(image_files, fps=1)
        save_dir = os.path.join(EXECUTOR_OUTPUT_DIR, save_dir_prefix)
        if not os.path.exists(save_dir):
                os.makedirs(save_dir,exist_ok=True)
        save_path = f'{save_dir}/{str(uuid.uuid4())}.mp4'
        clip.write_videofile(save_path, codec='libx264')
        logging.info(f"#4_1_3: write video : {save_path}")
        return {"image_paths":image_files, "video_path":f'{save_path}'}

    
    def clear_temp_python_files(self, generated_python_path_list):
        if generated_python_path_list:
            for path in generated_python_path_list:
                if os.path.isfile(path):
                    os.remove(path)
                    logging.info(f"Deleted：{path}")
                else:
                    logging.error(f"Not a file, can not delete：{path}")

    
    def execute_adapter_function(self, idx, function_body, arguments):
        logging.info("===execute_adapter_function===")
        logging.info(function_body)
        logging.info(arguments)
        
        func_str_fixed = (function_body.encode('utf-8')
                          .decode('unicode_escape')
                          .replace(os.getenv("RESERVED_PATH",""),
                                   EXECUTOR_OUTPUT_DIR)
                          )
        func_name = None
        for line in func_str_fixed.strip().split('\n'):
            line = line.strip()
            if line.startswith("def "):
                func_name = line.split('def ')[1].split('(')[0]
                break
        if not func_name:
            return (-1, {"error": "No function definition found"})

        local_scope = {}
        try:
            exec(func_str_fixed, local_scope, local_scope)
        except Exception as e:
            result = (-1, {"error": f"Function compilation error: {str(e)}"})

        target_func = local_scope.get(func_name)
        if not target_func:
            result = (-1, {"error": "Function definition not found"})

        try:
            result = (1, target_func(**arguments))
        except Exception as e:
            result = (-1, {"error": f"Function execution error: {str(e)}"})

        logging.info(result)
        logging.info("==============================")

        step_code, step_data = result

        if step_code == 1 and step_data is not None:
            step_data = {f'output_step_{idx}.{k}': v for k, v in step_data.items()}
        elif step_code == 1 and step_data is None:
            step_code = -1
            step_data = {f'output_step_{idx}.err_message':NONE_ERR_INFO}
        else:
            step_code = -1
            step_data = {f'output_step_{idx}.err_message':step_data}
        
        return step_code, step_data
    
    @timeout(TIMEOUT_SECONDS) 
    def run_pipeline(self, mode:int = 1, query:str = None, media:Union[str, List[str]] = None, error_history:list = None):
        if error_history is None:
            error_history = []
        generated_python_path_list = []
        
        logging.info("==========Run_pipeline==========")
        logging.info(query)
        logging.info("--------------------------------")
        logging.info(error_history)
        logging.info("--------------------------------")
        logging.info(self.global_result_dict)
        logging.info("--------------------------------")
        logging.info(self.global_response_str)
        logging.info("================================")

        logging.info("#1: Get Respons string from LLM.")
        if mode==1:
            if error_history == []:
                response_str, error_history = self.get_response_from_llm_based_on_schema(query = query)
            else:
                response_str, error_history = self.get_response_from_llm_based_on_schema(query = query, error_history = error_history)
        else:
            response_str = query

        logging.info("#1_1: Orignal response_str")
        logging.info(response_str)
        response_str = self.filter_llm_output(response_str)

        logging.info("#1_2: Filterded response_str")
        self.global_response_str = response_str
        logging.info(self.global_response_str)
        
        logging.info("#2: Trans LLM Respons to Dict.")
        response_dict = json.loads(response_str)
        logging.info(response_dict)
        
        logging.info("(cancel)#3: Prepear local python code.")


        logging.info("#4: Prepear local pipeline.")
        pipeline = response_dict["pipeline"]
        logging.info(pipeline)

        if mode in [1,2]:
            logging.info("#4_1: Prepear local media.")
            if error_history == []:
                if media is not None:
                    self.real_image_list = media
                    self.real_video_path = media
                elif media == "$search":
                    real_image_list, real_video_path = self.get_real_media(pipeline)
                    self.real_image_list = real_image_list
                    self.real_video_path = real_video_path
                elif media == "$static":
                    self.real_image_list = DEFAULT_IMAGE_LIST
                    self.real_video_path = DEFAULT_VIDEO_PATH
                else:
                    self.real_image_list = DEFAULT_IMAGE_LIST
                    self.real_video_path = DEFAULT_VIDEO_PATH

        logging.info("#5: execute pipeline.")
        for idx, step in enumerate(pipeline):
            # step = {
            #     "function_name" : "detection",
            #     "arguments" : {"image_path_list":user_image_path_list,"query_list": ["instruments","valves","pipes"]}
            # }
            logging.info(f"##Start PIPELINE STEP {idx+1}")
            logging.info(f"##5_1(In Loop): Orignal-->\n{step}")
            
            if mode in [1,2]:
                step = self.change_step_input(step)
                logging.info(f"##5_2(In Loop): after input change-->\n{step}")
            
            step = self.change_step_output(step)
            logging.info(f"##5_3(In Loop): after output change-->\n{step}")

            try: 
                
                if step["function"] in BASE_FUNCTIONS:
                    call_function_args = {
                        "function_name" : step["function"],
                        "arguments" : step["input"],
                        "only_local": True
                    }
                    logging.info(f"##5_4(In Loop): call function-->{call_function_args}")
                    step_code, step_data = self.call_function(idx+1, call_function_args)
                elif step["function"] not in BASE_FUNCTIONS and (step["function"]+".py") in response_dict["adapter_function"]:
                    call_function_args = {
                        "function_body" : response_dict["adapter_function"][step["function"]+".py"],
                        "arguments" : step["input"],
                    }
                    step_code, step_data = self.execute_adapter_function(idx+1, call_function_args["function_body"], call_function_args["arguments"])
                else:
                    raise NotImplementedError(f"No Function called")
            
            except Exception as e:
                logging.info("##5_4_1(In Loop, Error): clear_temp_python_files.")
                self.clear_temp_python_files(generated_python_path_list)
                logging.error(f"##5_4_2(In Loop,Error): call function error{str(e)}")
                raise RuntimeError(str(e))
            
            self.global_result_dict.update(step_data)
            
            logging.info(f"##5_5(In Loop):update_step_result_{idx+1}---->{self.global_result_dict}")
        
            if step_code == -1:
                logging.info(f"##5_6(In Loop):Raise Exception")
                logging.info("##5_6_1(In Loop, Error): clear_temp_python_files.")
                self.clear_temp_python_files(generated_python_path_list)
                raise RuntimeError(str(step_data))         
            
            logging.info(f"##5_7(In Loop):End step{idx+1}")
        
        logging.info("#6: clear_temp_python_files.")
        self.clear_temp_python_files(generated_python_path_list)
        logging.info("All Done.")

    
    def run_pipeline_error_handler(self, 
                                   id:int = -1, 
                                   query:str = None, 
                                   source:str = None, 
                                   media:Union[str, List[str]] = None,
                                   error_history:List = None, 
                                   mode=0, 
                                   expr_id = 0, 
                                   retry_limit = 0,
                                   ):
        
        try:
            if error_history is None:
                error_history = []
            self.global_response_str = ""
            self.global_result_dict = {}
            self.run_pipeline(mode = mode, query = query, media = media, error_history = error_history)
            train_seed_positive={"id": id, "input":query, "output":self.global_response_str, "trajectory":self.global_result_dict, "history":error_history, "source":source, "model":PLANNER_LLM_MODEL_NAME}
            self.write_jsonl(train_seed_positive, os.path.join(RESULT_PREFIX, f"positive_{expr_id}.jsonl"))
            return True, train_seed_positive
        except Exception as e:
            logging.error("######Exception########")
            if isinstance(e, json.JSONDecodeError):
                err_string = JSON_ERR_INFO + str(e)
            elif isinstance(e, RuntimeError):
                err_string = PLAN_RUN_ERR_INFO + e.args[0]
            elif isinstance(e, KeyError):
                err_string = KEYWORD_ERR_INFO + str(e)
            else:
                tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                detailed_error_message = ''.join(tb_str)
                detailed_error_message = detailed_error_message.replace("^","")
                err_string = UNDIFINED_ERR_INFO + detailed_error_message
            logging.error(f"######Exception########{len(error_history)/2}:::{retry_limit}")
            logging.error(err_string)
            logging.error("########################")
            if len(error_history)/2 < retry_limit:
                error_detail = [
                    {"role": "assistant", "content": self.global_response_str},
                    {"role": "user", "content": err_string}
                ]
                error_history += error_detail
                logging.error(f"Retrying for the {len(error_history)/2} time, maximum retries are {retry_limit}!")
                return self.run_pipeline_error_handler(
                    id = id, 
                    query = query, 
                    source = source, 
                    mode = mode,
                    expr_id = expr_id, 
                    retry_limit = retry_limit,
                    error_history = error_history,
                )

            else:
                train_seed_negetive={"id": id, "input":query, "output":self.global_response_str, "trajectory":self.global_result_dict, "history":error_history, "source":source, "model":PLANNER_LLM_MODEL_NAME}
                self.write_jsonl(train_seed_negetive, os.path.join(RESULT_PREFIX, f"negetive_{expr_id}.jsonl"))
                logging.error(MAXIMUM_RETRIES_REACHED_ERROR)
                return False, train_seed_negetive


def mode_switch(query = None, mode = 1, expr_id = 0, retry_limit = 0):
    # input ("id","query","source")
    start_time = time.time()
    if isinstance(query, str):
        with open(query, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif isinstance(query,list):
            data = query
    else:
        raise ValueError("Unsupported query data!")
    
    if data is None or len(data) == 0:
        raise ValueError("Data is none or empty!")
    
    if mode not in [1,2,3]:
        raise ValueError("Wrong mode number!")

    if mode!=1 and retry_limit!=0:
        retry_limit = 0
        print("Retry_limit only work in mode1!")

    

    success_case_list = []
    failed_case_list = []
    for item in data:
        if "query" in item:
            query = item.get('query')
        else:
            print("query is not in data")
            continue
        
        id = item.get('id') if "id" in item else "unbound_id"
        source = item.get('source') if "source" in item else "unbound_source"
        media = item.get('media') if "media" in item else None

        is_success, pipeline_result = AutoPipeline().run_pipeline_error_handler(
            id = id, 
            query = query, 
            source = source, 
            mode = mode,
            media = media,
            expr_id = expr_id, 
            retry_limit = retry_limit
            )
        
        if is_success:
            success_case_list.append(pipeline_result)
        else:
            failed_case_list.append(pipeline_result)

    success_pipeline_count = len(success_case_list)
    failed_pipeline_count = len(failed_case_list)
    total_pipeline_count = success_pipeline_count + failed_pipeline_count
    success_rate = success_pipeline_count / total_pipeline_count
    end_time = time.time()

    return {
        "total_pipeline_count":total_pipeline_count,
        "success_pipeline_count": success_pipeline_count,
        "failed_pipeline_count": failed_pipeline_count,
        "pipeline_success_rate": success_rate,
        "success_pipeline": success_case_list,
        "failed_pipeline": failed_case_list,
        "positive_pipeline_path": os.path.join(RESULT_PREFIX, f"positive_{expr_id}.jsonl"),
        "negetive_pipeline_path": os.path.join(RESULT_PREFIX, f"negetive_{expr_id}.jsonl"),
        "inference_time": end_time - start_time
        }