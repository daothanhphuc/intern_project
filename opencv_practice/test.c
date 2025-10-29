// #include <stdio.h>

// int main() {
//     char* s1 = "Hello"; //const char* s1 = "Hello"
//     // char s2[] = "Hello";
//     char* s3 = "Hello";
//     s3[3] = 'a'; //This line will cause undefined behavior
//     printf("s1: %s\n", s1);
//     printf("s3: %s\n", s3);
// }
// file: five_processes.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
// #include <sys/wait.h>
#include <windows.h>

int main(void) {
    int mynum = 1; // mặc định là parent sẽ in 1
    pid_t pid;
    
    // Parent sẽ tạo 4 child, được gán các số 2..5
    for (int i = 2; i <= 5; ++i) {
        pid = fork();
        if (pid < 0) {
            printf("Fork failed\n");
            return 0;
        }
        if (pid == 0) {
            // Đây là child mới: gán số và thoát vòng lặp để không fork tiếp
            mynum = i;
            break;
        }
    }

    printf("%d\n", mynum);
    // fflush(stdout);

    if (mynum == 1) {
        for (int j = 0; j < 4; ++j) {
            wait(NULL);
        }
    }
    return 0;
}
