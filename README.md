# MEC_DPC_CORAA
## Code Availability Notice
- This repository contains the implementation of the proposed method in the paper "[Distributed multi-agent reinforcement learning approach for multi-server multi-user task offloading]".
- **Temporary restrictions**: Some functions (marked with `# [RESTRICTED]` in the code) are commented to protect proprietary algorithms. Full code will be released after paper acceptance.

## Training Code
| File Path                             | Corresponding Section & Algorithm          | Description                                                              |
|---------------------------------------|-----------------------------|----------------------------------------------------------------------|
| `src/training/partial_commu.py`       | Section IV-B (Algorithm 1)  | Implementation of Partial Communication Model                   |
| `src/training/train_DPC_CORAA.py`     | Section IV-C (Algorithm 2)  | Training framework for DPC-CORAA (Distributed Policy Coordination and Resource Allocation)                         |
| `src/training/train_ABA.py`           | Section V-B (Baseline)       | Training implementation of ABA algorithm (baseline comparison)                   |
| `src/training/train_MADDPG_DTDE.py`   | Section V-B (Baseline)       | Training implementation of MADDPG with DTDE scheme (baseline comparison)    |
| `src/training/train_MADDPG_CTDE.py`   | Section V-D (Baseline)       | Training implementation of MADDPG with CTDE scheme   (baseline comparison)  |
| `src/training/train_MAPPO_CTDE.py`    | Section V-D (Baseline)       | Training implementation of MAPPO with CTDE scheme   (baseline comparison)                           |

## Testing Code
|File Path                                   |Corresponding Figures                 | Description                                                                 |
|-------------------------------------------|-----------------------------|----------------------------------------------------------------------|
| `src/testing/test_FLO.py`                 | Fig4, Fig5                  | Validates performance of FLO baseline                          |
| `src/testing/test_ROM.py`                 | Fig4, Fig5                  | Validates performance of ROM baseline                                 |
| `src/testing/test_ABA.py`                 | Fig4, Fig5                  | Validates performance of ABA baseline                                    |
| `src/testing/test_num_DPC_CORAA.py`       | Fig9-Fig14                 | Scalability testing of DPC-CORAA under varying UDs/ESs populations                            |
| `src/testing/test_num_MADDPG_CTDE.py`     | Fig9-Fig14                 | Scalability testing of MADDPG with CTDE scheme under varying UDs/ESs populations                         |
| `src/testing/test_num_MAPPO_CTDE.py`      | Fig9-Fig14                 | Scalability testing of MAPPO with CTDE scheme under varying UDs/ESs populations                           |
| `src/testing/test_DPC_CORAA.py`           | Fig15-Fig23                | Performance analysis of DPC-CORAA                    |
| `src/testing/test_MADDPG_CTDE.py`         | Fig15-Fig23                | Performance analysis of MADDPG with CTDE scheme                               |
| `src/testing/test_MAPPO_CTDE.py`          | Fig15-Fig23                | Performance analysis of MAPPO with CTDE scheme 
