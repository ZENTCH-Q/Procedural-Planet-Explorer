#include <cstdio>
#include <vector>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// — window size —
static const int WIN_W = 1280;
static const int WIN_H = 720;

// — sphere resolution —
static const int SPHERE_LAT = 128;
static const int SPHERE_LON = 256;

// — camera state —
float camYaw       = 45.0f,
      camPitch     = 30.0f,
      camDist      = 5.0f;
float targetYaw    = camYaw,
      targetPitch  = camPitch,
      targetDist   = camDist;
float yawVel       = 0.0f,
      pitchVel     = 0.0f,
      distVel      = 0.0f;

double lastFrameTime = 0.0;
const float ROT_SPEED    = 0.3f;
const float SCROLL_SPEED = 0.5f;
const float DAMPING      = 8.0f;
glm::vec3 camTarget(0.0f);

// — left‐drag: spin the planet; right‐drag: orbit camera —
bool leftDrag    = false,
     rightDrag   = false;
double lastX     = 0.0,
       lastY     = 0.0;
float sphereYaw      = 0.0f,
      spherePitch    = 0.0f;
float sphereYawVel   = 0.0f,
      spherePitchVel = 0.0f;

// — procedural & biome sliders —
struct Settings {
    float noiseFreq      = 1.0f;
    int   noiseOctaves   = 5;
    float noiseAmplitude = 0.1f;
    float waterLevel     = 0.30f;
    float beachLevel     = 0.32f;
    float desertThresh   = 0.30f;
    float forestThresh   = 0.60f;
    float snowLevel      = 0.80f;
} S;

// — Perlin noise (CPU) —
static int perm[512];
static void initNoise(){
    static const int p[256] = {
      151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
      140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
      247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
      57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
      74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
      60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
      65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
      200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
      52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
      207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
      119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
      129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
      218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
      81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
      184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
      222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };
    for(int i=0; i<256; ++i) perm[i] = perm[i+256] = p[i];
}
static float fadeF(float t){ return t*t*t*(t*(t*6-15)+10); }
static float lerpF(float a, float b, float t){ return a + t*(b - a); }
static float gradF(int h, float x, float y, float z){
    h &= 15;
    float u = h<8 ? x : y;
    float v = h<4 ? y : (h==12||h==14 ? x : z);
    return ((h&1)? -u : u) + ((h&2)? -v : v);
}
static float perlin3(float x, float y, float z){
    int X = int(floor(x)) & 255,
        Y = int(floor(y)) & 255,
        Z = int(floor(z)) & 255;
    x -= floor(x); y -= floor(y); z -= floor(z);
    float u = fadeF(x), v = fadeF(y), w = fadeF(z);
    int A  = perm[X] + Y, AA = perm[A] + Z, AB = perm[A+1] + Z;
    int B  = perm[X+1] + Y, BA = perm[B] + Z, BB = perm[B+1] + Z;
    return lerpF(
        lerpF(
            lerpF(gradF(perm[AA], x,   y,   z),
                  gradF(perm[BA], x-1, y,   z), u),
            lerpF(gradF(perm[AB], x,   y-1, z),
                  gradF(perm[BB], x-1, y-1, z), u), v),
        lerpF(
            lerpF(gradF(perm[AA+1], x,   y,   z-1),
                  gradF(perm[BA+1], x-1, y,   z-1), u),
            lerpF(gradF(perm[AB+1], x,   y-1, z-1),
                  gradF(perm[BB+1], x-1, y-1, z-1), u), v), w);
}

static void buildSphere(
    std::vector<glm::vec3>& pos,
    std::vector<glm::vec3>& nrm,
    std::vector<float>&    moist,
    std::vector<unsigned>& idx)
{
    initNoise();
    pos.clear(); nrm.clear(); moist.clear(); idx.clear();
    for(int y=0; y<=SPHERE_LAT; ++y){
        float v = float(y) / SPHERE_LAT;
        float phi = v * glm::pi<float>();
        for(int x=0; x<=SPHERE_LON; ++x){
            float u = float(x) / SPHERE_LON;
            float th = u * glm::two_pi<float>();
            glm::vec3 p = {
                sin(phi)*cos(th),
                cos(phi),
                sin(phi)*sin(th)
            };
            // fractal elevation
            float e = 0.0f;
            float amp = 1.0f;
            float freq = S.noiseFreq;
            for(int o=0; o<S.noiseOctaves; ++o){
                e += perlin3(p.x * freq, p.y * freq, p.z * freq) * amp;
                freq *= 2.0f;
                amp   *= 0.5f;
            }
            float radius = 1.0f + e * S.noiseAmplitude;
            glm::vec3 dp = p * radius;
            pos.push_back(dp);
            nrm.push_back(glm::normalize(dp));
            float m = perlin3((p.x+5)*3, (p.y+5)*3, (p.z+5)*3) * 0.5f + 0.5f;
            moist.push_back(glm::clamp(m, 0.0f, 1.0f));
        }
    }
    for(int y=0; y<SPHERE_LAT; ++y){
        for(int x=0; x<SPHERE_LON; ++x){
            int i0 = y*(SPHERE_LON+1) + x;
            int i1 = i0 + 1;
            int i2 = i0 + (SPHERE_LON+1);
            int i3 = i2 + 1;
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
        }
    }
}

// GLFW callbacks
static void error_cb(int e, const char* d){ fprintf(stderr, "GLFW Error %d: %s\n", e, d); }
static void fb_cb(GLFWwindow*, int w, int h){ glViewport(0, 0, w, h); }
static void mb_cb(GLFWwindow* w, int button, int action, int){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        leftDrag = (action == GLFW_PRESS);
        if(leftDrag) glfwGetCursorPos(w, &lastX, &lastY);
    } else if(button == GLFW_MOUSE_BUTTON_RIGHT){
        rightDrag = (action == GLFW_PRESS);
        if(rightDrag) glfwGetCursorPos(w, &lastX, &lastY);
    }
}
static void cur_cb(GLFWwindow* w, double x, double y){
    float dx = float(x - lastX);
    float dy = float(y - lastY);
    if(leftDrag){
        sphereYawVel   += dx * ROT_SPEED;
        spherePitchVel += dy * ROT_SPEED;
    } else if(rightDrag){
        yawVel   += dx * ROT_SPEED;
        pitchVel += dy * ROT_SPEED;
    }
    lastX = x; lastY = y;
}
static void sc_cb(GLFWwindow*, double, double yoff){
    distVel += float(yoff) * SCROLL_SPEED;
}

// Shader compilation
static GLuint compileShader(GLenum t, const char* s){
    GLuint sh = glCreateShader(t);
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if(!ok){ char buf[512]; glGetShaderInfoLog(sh, 512, nullptr, buf); fprintf(stderr, "Shader err: %s\n", buf); }
    return sh;
}
static GLuint createProgram(const char* vs, const char* fs){
    GLuint V = compileShader(GL_VERTEX_SHADER, vs);
    GLuint F = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint P = glCreateProgram();
    glAttachShader(P, V);
    glAttachShader(P, F);
    glLinkProgram(P);
    GLint linked; glGetProgramiv(P, GL_LINK_STATUS, &linked);
    if(!linked) fprintf(stderr, "Program link error\n");
    glDeleteShader(V);
    glDeleteShader(F);
    return P;
}

// GLSL sources
static const char* sphereVert = R"GLSL(
#version 450 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aMoist;
uniform mat4 uMVP;
out vec3 vNormal;
out vec3 vWorldPos;
out float vMoist;
void main(){ vNormal = aNormal; vWorldPos = aPos; vMoist = aMoist; gl_Position = uMVP * vec4(aPos,1.0); }
)GLSL";
static const char* sphereFrag = R"GLSL(
#version 450 core
in vec3 vNormal; in vec3 vWorldPos; in float vMoist;
out vec4 FragColor;
uniform vec3 uLightDir;
uniform float uWaterLevel, uBeachLevel, uDesertThresh, uForestThresh, uSnowLevel;
uniform vec3 uOceanColor, uBeachColor, uGrassColor, uForestColor, uDesertColor, uSnowColor;
void main(){
    float h = (length(vWorldPos) - 0.9) / 0.2;
    h = clamp(h, 0.0, 1.0);
    vec3 base;
    if(h < uWaterLevel) base = uOceanColor;
    else if(h < uBeachLevel){ float t = (h - uWaterLevel) / (uBeachLevel - uWaterLevel); base = mix(uOceanColor, uBeachColor, t);
    } else {
        if(vMoist < uDesertThresh) base = uDesertColor;
        else if(vMoist < uForestThresh) base = uGrassColor;
        else base = uForestColor;
        if(h > uSnowLevel) base = uSnowColor;
    }
    float Ndot = max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
    vec3 col = base * Ndot + 0.1 * base;
    FragColor = vec4(col,1.0);
}
)GLSL";

int main(){
    glfwSetErrorCallback(error_cb);
    if(!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,5);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    #endif
    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "Procedural Planet", nullptr, nullptr);
    if(!win) return -1;
    glfwMakeContextCurrent(win);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // callbacks
    glfwSetFramebufferSizeCallback(win, fb_cb);
    glfwSetMouseButtonCallback(win, mb_cb);
    glfwSetCursorPosCallback(win, cur_cb);
    glfwSetScrollCallback(win, sc_cb);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // build initial mesh
    std::vector<glm::vec3> positions, normals;
    std::vector<float> moisture;
    std::vector<unsigned> indices;
    buildSphere(positions, normals, moisture, indices);

    // upload buffers
    GLuint VAO, VBO, NBO, MBO, IBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &NBO);
    glGenBuffers(1, &MBO);
    glGenBuffers(1, &IBO);
    glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, positions.size()*sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);
      glBindBuffer(GL_ARRAY_BUFFER, NBO);
      glBufferData(GL_ARRAY_BUFFER, normals.size()*sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,(void*)0);
      glBindBuffer(GL_ARRAY_BUFFER, MBO);
      glBufferData(GL_ARRAY_BUFFER, moisture.size()*sizeof(float), moisture.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,0,(void*)0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    // shader program
    GLuint prog = createProgram(sphereVert, sphereFrag);
    glUseProgram(prog);
    glEnable(GL_DEPTH_TEST);

    // static uniforms
    glUniform3f(glGetUniformLocation(prog,"uOceanColor"),  0.0f,0.0f,0.3f);
    glUniform3f(glGetUniformLocation(prog,"uBeachColor"),  0.8f,0.7f,0.4f);
    glUniform3f(glGetUniformLocation(prog,"uGrassColor"),  0.1f,0.6f,0.2f);
    glUniform3f(glGetUniformLocation(prog,"uForestColor"), 0.0f,0.4f,0.0f);
    glUniform3f(glGetUniformLocation(prog,"uDesertColor"), 0.8f,0.8f,0.2f);
    glUniform3f(glGetUniformLocation(prog,"uSnowColor"),   1.0f,1.0f,1.0f);
    glUniform3f(glGetUniformLocation(prog,"uLightDir"),   0.5f,1.0f,0.3f);

    // caching last noise settings
    float lastFreq = S.noiseFreq;
    int   lastOct  = S.noiseOctaves;
    float lastAmp  = S.noiseAmplitude;

    lastFrameTime = glfwGetTime();
    while(!glfwWindowShouldClose(win)){
        // timing
        double now = glfwGetTime();
        float  dt  = float(now - lastFrameTime);
        lastFrameTime = now;

        // integrate camera
        targetYaw   += yawVel * dt;
        targetPitch += pitchVel * dt;
        targetDist  += distVel * dt;
        targetPitch = glm::clamp(targetPitch, 5.0f, 175.0f);
        targetDist  = glm::clamp(targetDist, 1.0f, 50.0f);
        float damp = expf(-DAMPING * dt);
        yawVel    *= damp;
        pitchVel  *= damp;
        distVel   *= damp;
        float smooth = 1.0f - damp;
        camYaw   = glm::mix(camYaw,   targetYaw,   smooth);
        camPitch = glm::mix(camPitch, targetPitch, smooth);
        camDist  = glm::mix(camDist,  targetDist,  smooth);

        // integrate sphere spin
        sphereYaw   += sphereYawVel   * dt;
        spherePitch = glm::clamp(spherePitch + spherePitchVel * dt, -89.0f, 89.0f);
        sphereYawVel   *= damp;
        spherePitchVel *= damp;

        // events
        glfwPollEvents();
        
        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Planet Settings");
          ImGui::SliderFloat("Noise Freq", &S.noiseFreq, 0.1f, 4.0f);
          ImGui::SliderInt  ("Octaves",    &S.noiseOctaves, 1, 8);
          ImGui::SliderFloat("Noise Amp",  &S.noiseAmplitude, 0.01f, 0.3f);
          ImGui::Separator();
          ImGui::SliderFloat("Water Level",   &S.waterLevel,   0, 1);
          ImGui::SliderFloat("Beach Level",   &S.beachLevel,   S.waterLevel, 1);
          ImGui::SliderFloat("Desert Moisture", &S.desertThresh, 0, 1);
          ImGui::SliderFloat("Forest Moisture", &S.forestThresh, 0, 1);
          ImGui::SliderFloat("Snow Level",      &S.snowLevel,    0, 1);
        ImGui::End();
        ImGui::Render();

        // rebuild mesh if noise changed
        if(lastFreq != S.noiseFreq || lastOct != S.noiseOctaves || lastAmp != S.noiseAmplitude){
            buildSphere(positions, normals, moisture, indices);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size()*sizeof(glm::vec3), positions.data());
            glBindBuffer(GL_ARRAY_BUFFER, NBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size()*sizeof(glm::vec3), normals.data());
            glBindBuffer(GL_ARRAY_BUFFER, MBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, moisture.size()*sizeof(float), moisture.data());
            lastFreq = S.noiseFreq;
            lastOct  = S.noiseOctaves;
            lastAmp  = S.noiseAmplitude;
        }

        // biome uniforms
        glUniform1f(glGetUniformLocation(prog, "uWaterLevel"),   S.waterLevel);
        glUniform1f(glGetUniformLocation(prog, "uBeachLevel"),   S.beachLevel);
        glUniform1f(glGetUniformLocation(prog, "uDesertThresh"), S.desertThresh);
        glUniform1f(glGetUniformLocation(prog, "uForestThresh"), S.forestThresh);
        glUniform1f(glGetUniformLocation(prog, "uSnowLevel"),    S.snowLevel);

        // matrices
        int w,h; glfwGetFramebufferSize(win, &w, &h);
        float aspect = float(w)/float(h);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
        glm::vec3 camOff = {
            camDist * cos(glm::radians(camPitch)) * cos(glm::radians(camYaw)),
            camDist * sin(glm::radians(camPitch)),
            camDist * cos(glm::radians(camPitch)) * sin(glm::radians(camYaw))
        };
        glm::mat4 view = glm::lookAt(camTarget + camOff, camTarget, {0,1,0});
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(sphereYaw), {0,1,0});
        model = glm::rotate(model, glm::radians(spherePitch), {1,0,0});
        glm::mat4 mvp = proj * view * model;
        glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));

        // render
        glViewport(0,0,w,h);
        glClearColor(0.05f,0.05f,0.1f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &NBO);
    glDeleteBuffers(1, &MBO);
    glDeleteBuffers(1, &IBO);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
