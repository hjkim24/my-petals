#!/usr/bin/env python3
"""
모든 stage를 자동으로 실행하는 스크립트
사용법: python run_all.py [--model MODEL] [--splits SPLITS] [--max_tokens N] [--prompt PROMPT]
"""

import subprocess
import sys
import time
import re
import signal
import argparse
import os
import threading
import queue
from pathlib import Path


def get_local_ip():
    """로컬 IP 주소 가져오기"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def extract_dht_maddr(line):
    """로그 라인에서 DHT multiaddr 추출"""
    # 형식 1: "DHT visible multiaddrs (use for --dht_initial_peers): [<Multiaddr /ip4/...>]"
    match = re.search(r"<Multiaddr\s+([^>]+)>", line)
    if match:
        return match.group(1).strip()
    
    # 형식 2: "DHT visible multiaddrs (use for --dht_initial_peers): ['/ip4/...']"
    match = re.search(r"DHT visible multiaddrs.*?:\s*\[?['\"]([^'\"]+)['\"]", line)
    if match:
        return match.group(1)
    
    # 형식 3: "try fallback: ['/ip4/...']"
    match = re.search(r"try fallback:.*?\[['\"]([^'\"]+)['\"]", line)
    if match:
        return match.group(1)
    
    return None


def check_stage_ready(line, stage_num):
    """로그 라인에서 stage가 준비되었는지 확인"""
    if stage_num in [1, 2, 3]:
        # 서버 stage: "handlers registered, waiting for requests..." 메시지 확인
        # 패턴: "Stage{num} handlers registered" 또는 "handlers registered.*waiting"
        if re.search(rf"Stage{stage_num}\s+handlers registered", line, re.IGNORECASE):
            return True
        if re.search(r"handlers registered.*waiting", line, re.IGNORECASE):
            return True
        # 더 넓은 패턴: "handlers registered"만 있어도 OK
        if re.search(r"handlers registered", line, re.IGNORECASE):
            return True
    elif stage_num == 0:
        # 클라이언트 stage: "RpcTransport initialized" 메시지 확인
        if re.search(r"RpcTransport initialized", line, re.IGNORECASE):
            return True
        # DHT 초기화 후에도 준비된 것으로 간주
        if re.search(r"Initializing DHT", line, re.IGNORECASE):
            return True
    return False


def wait_for_stage_ready(process, stage_name, stage_num, log_handle=None, quiet=False, timeout=60):
    """Stage가 준비될 때까지 대기"""
    ready_queue = queue.Queue()
    ready_event = threading.Event()
    
    def read_output():
        """출력을 읽으면서 준비 신호 확인"""
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                ready_queue.put(('stdout', line))
                if log_handle:
                    log_handle.write(line)
                    log_handle.flush()
                
                # 준비 신호 확인 (디버깅용으로 로그도 출력)
                if check_stage_ready(line, stage_num):
                    if not ready_event.is_set():
                        ready_event.set()
                        if not quiet:
                            print(f"\n✓ {stage_name} 준비 완료! (신호: {line.strip()})")
        except Exception as e:
            if not quiet:
                print(f"[{stage_name}] read_output 오류: {e}")
            pass
    
    def read_stderr():
        """stderr 읽기"""
        try:
            for line in iter(process.stderr.readline, ''):
                if not line:
                    break
                ready_queue.put(('stderr', line))
                if log_handle:
                    log_handle.write(line)
                    log_handle.flush()
                
                # 준비 신호 확인
                if check_stage_ready(line, stage_num):
                    if not ready_event.is_set():
                        ready_event.set()
                        if not quiet:
                            print(f"\n✓ {stage_name} 준비 완료! (신호: {line.strip()})")
        except Exception as e:
            if not quiet:
                print(f"[{stage_name}] read_stderr 오류: {e}")
            pass
    
    stdout_thread = threading.Thread(target=read_output, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    
    # 준비 신호 대기 (최대 timeout초)
    for i in range(timeout * 2):  # 0.5초 간격
        if process.poll() is not None:
            if not quiet:
                print(f"\n오류: {stage_name}이 예기치 않게 종료되었습니다. 반환 코드: {process.returncode}")
            return False
        
        if ready_event.is_set():
            # 큐에 남은 출력 처리
            while not ready_queue.empty():
                try:
                    stream_type, line = ready_queue.get_nowait()
                    if not quiet:
                        prefix = f"[{stage_name}]" if stream_type == 'stdout' else f"[{stage_name}:ERR]"
                        print(f"{prefix} {line}", end='')
                except queue.Empty:
                    break
            return True
        
        # 큐에서 출력 표시
        try:
            stream_type, line = ready_queue.get(timeout=0.5)
            if not quiet:
                prefix = f"[{stage_name}]" if stream_type == 'stdout' else f"[{stage_name}:ERR]"
                print(f"{prefix} {line}", end='')
        except queue.Empty:
            if i % 10 == 0 and not quiet:  # 5초마다 진행 상황 출력
                print(f"[{stage_name} 준비 대기 중... {i*0.5:.1f}초 경과]")
            pass
    
    if not quiet:
        print(f"\n경고: {stage_name} 준비 신호를 {timeout}초 내에 받지 못했습니다. 계속 진행합니다...")
    return False


def run_stage(stage_num, model, splits, dht_maddr, dht_port, rpc_port, 
               prompt=None, max_tokens=None, log_file=None):
    """Stage 실행"""
    # 프로젝트 루트로 이동
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    cmd = [
        sys.executable, "-m", "src.main",
        "--model", model,
        "--splits", splits,
        "--stage", str(stage_num),
        "--dht_port", str(dht_port),
        "--rpc_port", str(rpc_port),
    ]
    
    if dht_maddr:
        cmd.extend(["--dht_initial_peers", dht_maddr])
    else:
        cmd.extend(["--dht_initial_peers", ""])
    
    if stage_num == 0:
        if prompt:
            cmd.extend(["--prompt", prompt])
        if max_tokens:
            cmd.extend(["--max_new_tokens", str(max_tokens)])
    
    # 로그 파일 열기 (버퍼링 없이)
    if log_file:
        log_handle = open(log_file, 'w', buffering=1)  # line buffering
    else:
        log_handle = None
    
    # stdout과 stderr을 모두 캡처하되, 로그 파일에도 기록
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line buffering
        universal_newlines=True
    )
    
    return process, log_handle


def read_process_output(process, stage_name, log_handle=None, quiet=False):
    """프로세스의 stdout과 stderr을 읽어서 출력 (스레드 안전)"""
    output_queue = queue.Queue()
    stop_event = threading.Event()
    
    def read_stdout():
        try:
            for line in iter(process.stdout.readline, ''):
                if stop_event.is_set():
                    break
                if line:
                    output_queue.put(('stdout', line))
        except Exception:
            pass
    
    def read_stderr():
        try:
            for line in iter(process.stderr.readline, ''):
                if stop_event.is_set():
                    break
                if line:
                    output_queue.put(('stderr', line))
        except Exception:
            pass
    
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    
    while process.poll() is None or not output_queue.empty():
        try:
            stream_type, line = output_queue.get(timeout=0.1)
            if not quiet:
                formatted_line = f"[{stage_name}] {line}"
                print(formatted_line, end='', flush=True)
            if log_handle:
                log_handle.write(line)  # 원본 라인만 로그 파일에 저장
                log_handle.flush()
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        except Exception as e:
            if not quiet:
                print(f"[{stage_name}] Error: {e}\n", end='', flush=True)
    
    stop_event.set()
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description="모든 stage를 자동으로 실행")
    parser.add_argument("--model", default="gpt2", help="모델 이름")
    parser.add_argument("--splits", default="10,20,30", help="Splits (예: 10,20,30)")
    parser.add_argument("--max_tokens", type=int, default=32, help="최대 생성 토큰 수")
    parser.add_argument("--prompt", default="Hello, how are you?", help="입력 프롬프트")
    parser.add_argument("--dht_base_port", type=int, default=8000, help="DHT 기본 포트")
    parser.add_argument("--rpc_base_port", type=int, default=8001, help="RPC 기본 포트")
    parser.add_argument("--quiet", action="store_true", help="콘솔 출력 없이 로그 파일만 사용")
    
    args = parser.parse_args()
    
    local_ip = get_local_ip()
    processes = []
    log_handles = []
    dht_maddr = None
    
    def cleanup():
        """모든 프로세스 종료"""
        print("\n모든 stage 종료 중...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
        for h in log_handles:
            if h:
                h.close()
    
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    
    try:
        # Stage1 시작
        print("=" * 50)
        print("Stage1 시작 중...")
        print("=" * 50)
        stage1, log1 = run_stage(
            1, args.model, args.splits, None,
            args.dht_base_port, args.rpc_base_port,
            log_file="stage1.log"
        )
        processes.append(stage1)
        if log1:
            log_handles.append(log1)
        
        # DHT multiaddr 추출 (최대 30초 대기)
        if not args.quiet:
            print("Stage1 DHT 초기화 대기 중...")
        
        stage1_queue = queue.Queue()
        dht_found = threading.Event()
        found_maddr = [None]  # 리스트로 감싸서 nonlocal 효과
        
        def read_stage1_stdout():
            """Stage1의 stdout을 읽어서 큐에 넣고 DHT multiaddr 찾기"""
            try:
                for line in iter(stage1.stdout.readline, ''):
                    if not line:
                        break
                    stage1_queue.put(('stdout', line))
                    if log1:
                        log1.write(line)
                        log1.flush()
                    
                    maddr = extract_dht_maddr(line)
                    if maddr and not dht_found.is_set():
                        found_maddr[0] = maddr
                        dht_found.set()
                        if not args.quiet:
                            print(f"\n✓ DHT multiaddr 발견: {found_maddr[0]}")
            except Exception as e:
                stage1_queue.put(('error', f"Error reading stdout: {e}\n"))
        
        def read_stage1_stderr():
            """Stage1의 stderr을 읽어서 큐에 넣고 DHT multiaddr 찾기"""
            try:
                for line in iter(stage1.stderr.readline, ''):
                    if not line:
                        break
                    stage1_queue.put(('stderr', line))
                    if log1:
                        log1.write(line)
                        log1.flush()
                    
                    maddr = extract_dht_maddr(line)
                    if maddr and not dht_found.is_set():
                        found_maddr[0] = maddr
                        dht_found.set()
                        if not args.quiet:
                            print(f"\n✓ DHT multiaddr 발견: {found_maddr[0]}")
            except Exception as e:
                stage1_queue.put(('error', f"Error reading stderr: {e}\n"))
        
        # Stage1이 시작될 때까지 잠시 대기
        time.sleep(1)
        
        # Stage1 프로세스 상태 확인
        if stage1.poll() is not None:
            if not args.quiet:
                print(f"오류: Stage1이 즉시 종료되었습니다. 반환 코드: {stage1.returncode}")
                # stderr에서 에러 메시지 읽기 시도
                try:
                    err_output = stage1.stderr.read()
                    if err_output:
                        print(f"에러 출력:\n{err_output.decode('utf-8', errors='ignore')}")
                except:
                    pass
            cleanup()
            sys.exit(1)
        
        stdout_thread = threading.Thread(target=read_stage1_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stage1_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # DHT multiaddr 대기 (최대 30초)
        for i in range(60):  # 0.5초 간격으로 60번 = 30초
            if stage1.poll() is not None:
                if not args.quiet:
                    print(f"\n오류: Stage1이 예기치 않게 종료되었습니다. 반환 코드: {stage1.returncode}")
                    # 남은 에러 메시지 출력
                    try:
                        err_output = stage1.stderr.read()
                        if err_output:
                            print(f"에러 출력:\n{err_output.decode('utf-8', errors='ignore')}")
                    except:
                        pass
                cleanup()
                sys.exit(1)
            
            if dht_found.is_set():
                # 큐에서 남은 라인들 처리
                while not stage1_queue.empty():
                    try:
                        stream_type, line = stage1_queue.get_nowait()
                        if not args.quiet:
                            prefix = "[Stage1]" if stream_type == 'stdout' else "[Stage1:ERR]"
                            print(f"{prefix} {line}", end='')
                    except queue.Empty:
                        break
                break
            
            # 큐에서 라인 출력
            try:
                stream_type, line = stage1_queue.get(timeout=0.5)
                if not args.quiet:
                    prefix = "[Stage1]" if stream_type == 'stdout' else "[Stage1:ERR]"
                    print(f"{prefix} {line}", end='')
            except queue.Empty:
                if i % 10 == 0 and not args.quiet:  # 5초마다 진행 상황 출력
                    print(f"[대기 중... {i*0.5:.1f}초 경과]")
                pass
        
        if not dht_found.is_set() or not found_maddr[0]:
            if not args.quiet:
                print("오류: Stage1에서 DHT multiaddr을 찾을 수 없습니다.")
            cleanup()
            sys.exit(1)
        
        dht_maddr = found_maddr[0]
        
        # Stage1이 완전히 준비될 때까지 대기
        # 이미 읽은 라인들 중에서도 준비 신호 확인
        if not args.quiet:
            print("\nStage1 완전 초기화 대기 중...")
        
        # 이미 읽은 큐에서 준비 신호 확인
        stage1_ready = False
        temp_queue = queue.Queue()
        while not stage1_queue.empty():
            try:
                stream_type, line = stage1_queue.get_nowait()
                temp_queue.put((stream_type, line))
                if check_stage_ready(line, 1):
                    stage1_ready = True
                    if not args.quiet:
                        print(f"\n✓ Stage1 준비 완료! (이미 읽은 메시지에서 발견)")
                    break
            except queue.Empty:
                break
        
        # 큐 복원
        while not temp_queue.empty():
            stage1_queue.put(temp_queue.get())
        
        # 아직 준비 신호를 못 찾았다면 추가로 대기
        if not stage1_ready:
            # 기존 스레드가 계속 읽고 있으므로, 추가로 읽은 라인 확인
            for i in range(60):  # 최대 30초
                if stage1.poll() is not None:
                    break
                
                try:
                    stream_type, line = stage1_queue.get(timeout=0.5)
                    if not args.quiet:
                        prefix = "[Stage1]" if stream_type == 'stdout' else "[Stage1:ERR]"
                        print(f"{prefix} {line}", end='')
                    
                    if check_stage_ready(line, 1):
                        stage1_ready = True
                        if not args.quiet:
                            print(f"\n✓ Stage1 준비 완료!")
                        break
                except queue.Empty:
                    if i % 10 == 0 and not args.quiet:
                        print(f"[Stage1 준비 대기 중... {i*0.5:.1f}초 경과]")
                    pass
        
        if not stage1_ready and not args.quiet:
            print("경고: Stage1 준비 신호를 받지 못했지만 계속 진행합니다...")
        
        # 남은 Stage1 출력 처리
        stage1_reader_thread = threading.Thread(
            target=lambda: read_process_output(stage1, "Stage1", log1, args.quiet),
            daemon=True
        )
        stage1_reader_thread.start()
        
        # Stage2 시작
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Stage2 시작 중...")
            print("=" * 50)
        stage2, log2 = run_stage(
            2, args.model, args.splits, dht_maddr,
            args.dht_base_port + 2, args.rpc_base_port + 2,
            log_file="stage2.log"
        )
        processes.append(stage2)
        if log2:
            log_handles.append(log2)
        
        # Stage2가 완전히 준비될 때까지 대기
        if not args.quiet:
            print("\nStage2 완전 초기화 대기 중...")
        if not wait_for_stage_ready(stage2, "Stage2", 2, log2, args.quiet, timeout=30):
            if not args.quiet:
                print("경고: Stage2 준비 신호를 받지 못했지만 계속 진행합니다...")
        
        # Stage2 로그 출력 (스레드로)
        stage2_thread = threading.Thread(
            target=lambda: read_process_output(stage2, "Stage2", log2, args.quiet),
            daemon=True
        )
        stage2_thread.start()
        
        # Stage3 시작
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Stage3 시작 중...")
            print("=" * 50)
        stage3, log3 = run_stage(
            3, args.model, args.splits, dht_maddr,
            args.dht_base_port + 4, args.rpc_base_port + 4,
            log_file="stage3.log"
        )
        processes.append(stage3)
        if log3:
            log_handles.append(log3)
        
        # Stage3가 완전히 준비될 때까지 대기
        if not args.quiet:
            print("\nStage3 완전 초기화 대기 중...")
        if not wait_for_stage_ready(stage3, "Stage3", 3, log3, args.quiet, timeout=30):
            if not args.quiet:
                print("경고: Stage3 준비 신호를 받지 못했지만 계속 진행합니다...")
        
        # Stage3 로그 출력 (스레드로)
        stage3_thread = threading.Thread(
            target=lambda: read_process_output(stage3, "Stage3", log3, args.quiet),
            daemon=True
        )
        stage3_thread.start()
        
        # Stage0 시작 (포그라운드)
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Stage0 시작 중...")
            print("=" * 50)
        stage0, log0 = run_stage(
            0, args.model, args.splits, dht_maddr,
            args.dht_base_port + 6, args.rpc_base_port + 6,
            prompt=args.prompt, max_tokens=args.max_tokens,
            log_file="stage0.log"
        )
        processes.append(stage0)
        if log0:
            log_handles.append(log0)
        
        # Stage0 출력을 실시간으로 표시 (메인 스레드에서)
        read_process_output(stage0, "Stage0", log0, args.quiet)
        
        # Stage0 종료 대기
        stage0.wait()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()