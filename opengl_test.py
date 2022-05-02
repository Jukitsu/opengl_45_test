import pyglet
import ctypes
import glm
from collections import deque
import math
from random import randint

pyglet.options["shadow_window"] = False
pyglet.options["debug_gl"] = False

from OpenGL.GL import *
from OpenGL.GL import shaders
config = pyglet.gl.Config(double_buffer = True,
				major_version = 4, minor_version = 6, depth_size = 24)
window = pyglet.window.Window(config = config, vsync = False, width = 852, height = 480, resizable = True)
fps_display = pyglet.window.FPSDisplay(window)

dim = [852, 480]


quad = [
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,

    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0
]


indices = [
	 0,  1,  2,  0,  2,  3, # right
	 4,  5,  6,  4,  6,  7, # left
	 8,  9, 10,  8, 10, 11, # top
	12, 13, 14, 12, 14, 15, # bottom
	16, 17, 18, 16, 18, 19, # front
	20, 21, 22, 20, 22, 23, # back
]

vertices = [
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    1.0,  0.0,  0.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 0.0,    1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    1.0,  0.0,  0.0,
     0.5,  0.5, -0.5,   1.0, 1.0, 0.0,    1.0,  0.0,  0.0,

    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,   -1.0,  0.0,  0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,   -1.0,  0.0,  0.0,

     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    0.0,  1.0,  0.0,
     0.5,  0.5, -0.5,   0.0, 0.0, 0.0,    0.0,  1.0,  0.0,
    -0.5,  0.5, -0.5,   1.0, 0.0, 0.0,    0.0,  1.0,  0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,    0.0,  1.0,  0.0,

    -0.5, -0.5,  0.5,   0.0, 1.0, 0.0,    0.0, -1.0,  0.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, 0.0,    0.0, -1.0,  0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,   1.0, 1.0, 0.0,    0.0, -1.0,  0.0,

    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,    0.0,  0.0,  1.0,
    -0.5, -0.5,  0.5,   0.0, 0.0, 0.0,    0.0,  0.0,  1.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,    0.0,  0.0,  1.0, 
     0.5,  0.5,  0.5,   1.0, 1.0, 0.0,    0.0,  0.0,  1.0,

     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,    0.0,  0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, 0.0,    0.0,  0.0, -1.0,
    -0.5, -0.5, -0.5,   1.0, 0.0, 0.0,    0.0,  0.0, -1.0,
    -0.5,  0.5, -0.5,   1.0, 1.0, 0.0,    0.0,  0.0, -1.0
]

instanced_array = []
instances = []
for i in range(-32, 32):
    for k in range(-32, 32):
        instanced_array.append((i, randint(0, 5), k, 0))

for i in range(-64, 64):
    for k in range(-64, 64):
        instanced_array.append((i, 0, k, 0))


for _ in range(256):
    instanced_array.append(((256 - randint(0, 512)) / 8, randint(0, 256) / 8, (256 - randint(0, 512)) / 8, 1))

light_instance = (200, 100, 200, 2)
instanced_array.append(light_instance)

cmd = [
    len(indices), len(instanced_array) - 1, 0, 0, 0,
    len(indices), 1, 0, 0, len(instanced_array) - 1
]

shadow_vert = """
#version 460 core

layout(location = 0) in vec3 a_VertexPosition;

layout(location = 3) in vec3 a_InstanceOffset;


layout(std140, binding = 1) uniform u_Lights {
    vec3 u_LightPos;
    float u_AmbientLight;
    float u_SpecularStrength;
    float u_Constant;
    float u_Linear;
    float u_Quadratic;
};


layout(location = 2) uniform mat4 u_LightViewProj;

void main(void) {
    gl_Position = u_LightViewProj * vec4(a_VertexPosition + a_InstanceOffset, 1.0);
}
"""
shadow_frag = """
#version 460 core

void main(void) {
    
}

"""
vert = """
#version 460 core

layout(location = 0) in vec3 a_VertexPosition;
layout(location = 1) in vec3 a_VertexTexCoords;
layout(location = 2) in vec3 a_VertexNormal;
layout(location = 3) in vec3 a_InstanceOffset;
layout(location = 4) in float a_InstanceTexIndex;

layout(std140, binding = 0) uniform u_Camera {
    mat4 u_ModelViewProj;
    vec3 u_CameraPos;
};
layout(std140, binding = 1) uniform u_Lights {
    vec3 u_LightPos;
    float u_AmbientLight;
    float u_SpecularStrength;
    float u_Constant;
    float u_Linear;
    float u_Quadratic;
};

layout(location = 2) uniform mat4 u_LightViewProj;

out vec3 v_Position;
out vec3 v_TexCoords;
out flat vec3 v_Normal;
out flat int v_Emmissive; // for light blocks, using push constants
out vec4 v_LightSpacePosition;

void main(void) {
    v_Position = a_VertexPosition + a_InstanceOffset;
    v_TexCoords = vec3(a_VertexTexCoords.xy, a_VertexTexCoords.z + a_InstanceTexIndex);
    v_Normal = a_VertexNormal;
    v_Emmissive = gl_DrawID; // Push constant
    v_LightSpacePosition = u_LightViewProj * vec4(v_Position, 1.0f);
    gl_Position = u_ModelViewProj * vec4(v_Position, 1.0f);
}
"""

frag = """
#version 460 core

in vec3 v_Position;
in vec3 v_TexCoords;
in flat vec3 v_Normal;
in flat int v_Emmissive;
in vec4 v_LightSpacePosition;

layout(location = 0) uniform sampler2DArray u_TextureArraySampler;
layout(location = 1) uniform sampler2DShadow u_ShadowMapSampler;

layout(std140, binding = 0) uniform u_Camera {
    mat4 u_ModelViewProj;
    vec3 u_CameraPos;
};
layout(std140, binding = 1) uniform u_Lights {
    vec3 u_LightPos;
    float u_AmbientLight;
    float u_SpecularStrength;
    float u_Constant;
    float u_Linear;
    float u_Quadratic;
};

#define SAMPLE_SIZE 4
#define SAMPLE_COUNT ((2 * SAMPLE_SIZE + 1) * (2 * SAMPLE_SIZE + 1))
layout(location = 2) uniform mat4 u_LightViewProj;

layout(location = 0) out vec4 fragColor;

float computeShadow(vec4 lightSpacePos, float diffuse) {
    vec3 projCoords = (lightSpacePos.xyz / lightSpacePos.w) * 0.5f + 0.5f;
    float currentDepth = projCoords.z;
    float bias = max(0.025f * (1.0f - diffuse), 0.0005f);
    float shadow = 0.0f;

    vec2 pixelSize = 1.0f / textureSize(u_ShadowMapSampler, 0);


    for (int y = -SAMPLE_SIZE; y <= SAMPLE_SIZE; y++)
        for (int x = -SAMPLE_SIZE; x <= SAMPLE_SIZE; x++) {
            float closestDepth = texture(u_ShadowMapSampler, projCoords + vec3(vec2(x, y) * pixelSize, 0));
            if (currentDepth > closestDepth + bias)
                shadow += 1.0f;
        }

    return 0.5f + shadow / (2 * SAMPLE_COUNT);
}

void main(void) {
    float ambientLight = u_AmbientLight;
    vec3 normal = normalize(v_Normal);
    vec3 lightRay = normalize(u_LightPos);
    float diffuseLight = max(dot(normal, lightRay), 0.0f);

    vec3 viewRay = normalize(u_CameraPos - v_Position);
    vec3 reflectRay = reflect(-lightRay, normal);
    float specularLight = u_SpecularStrength * pow(max(dot(viewRay, reflectRay), 0.0f), 32);
    float shadow = computeShadow(v_LightSpacePosition, diffuseLight);

    vec4 lightColor = vec4(1.0f);
    if (v_Emmissive != 0) {
        fragColor = lightColor;
    } else {
        fragColor = (ambientLight + (1.0 - shadow) * (diffuseLight + specularLight)) * lightColor * pow(texture(u_TextureArraySampler, v_TexCoords), vec4(2.2f));
    }
}
"""

ppvert = """
#version 460 core

layout(location = 0) in vec2 a_Position;
layout(location = 1) in vec2 a_TexCoords;

out vec2 v_Position;
out vec2 v_TexCoords;

void main(void) {
    v_Position = a_Position;
    v_TexCoords = a_TexCoords;
    gl_Position = vec4(a_Position, 1.0, 1.0);
}
"""
ppfrag = """
#version 460 core

in vec2 v_Position;
in vec2 v_TexCoords;

layout(location = 0) uniform sampler2D u_ColorBufferSampler;
layout(location = 1) uniform float exposure;
out vec4 fragColor;

void main(void) {
    vec3 hdrColor = texture(u_ColorBufferSampler, v_TexCoords).xyz;
    vec3 mappedColor = vec3(1.0) - exp(-hdrColor * exposure);
    float vignette = smoothstep(0.8, 0.0, 0.5 * distance(vec2(0.0), v_Position));
    fragColor = pow(vec4(mappedColor, 1.0f) * vignette, vec4(1.0f / 2.2f));
}
"""
shadow_vsh = shaders.compileShader(shadow_vert, GL_VERTEX_SHADER)
shadow_fsh = shaders.compileShader(shadow_frag, GL_FRAGMENT_SHADER)

shadow_program = shaders.compileProgram(shadow_vsh, shadow_fsh, validate=True)

vsh = shaders.compileShader(vert, GL_VERTEX_SHADER)
fsh = shaders.compileShader(frag, GL_FRAGMENT_SHADER)

program = shaders.compileProgram(vsh, fsh, validate=False)

ppvsh = shaders.compileShader(ppvert, GL_VERTEX_SHADER)
ppfsh = shaders.compileShader(ppfrag, GL_FRAGMENT_SHADER)
pprcs_program = shaders.compileProgram(ppvsh, ppfsh, validate=True)

vao = GLuint(0)
vbo = GLuint(0)
instance_sbo = GLuint(0)
instance_vbo = GLuint(0)
ibo = GLuint(0)
icbo = GLuint(0)
tao = GLuint(0)
msaa_fbo = GLuint(0)
msaa_cbo = GLuint(0)
msaa_dbo = GLuint(0)
pp_fbo = GLuint(0)
pp_cbo = GLuint(0)
pp_dbo = GLuint(0)
shadow_fbo = GLuint(0)
shadow_map = GLuint(0)
quad_vao = GLuint(0)
quad_vbo = GLuint(0)
mat_ubo = GLuint(0)
light_ubo = GLuint(0)
mapped_ubo = [None, None]
fences = deque()

camera_position = glm.vec3(0, 0, 3)
camera_rotation = [-math.tau / 4, 0.0]

user_input = [0, 0, 0]
mouse_captured = [True]

class Frustum:
    left = glm.vec4(1.0)
    right = glm.vec4(1.0)
    top = glm.vec4(1.0)
    bottom = glm.vec4(1.0)
    near = glm.vec4(1.0)
    far = glm.vec4(1.0)

def check_in_frustum(instance):
    """Frustum check of each chunk. If the chunk is not in the view frustum, it is discarded"""
    planes = (Frustum.left, Frustum.right, Frustum.bottom, Frustum.top, Frustum.near, Frustum.far)
    pos = glm.vec3(instance[0], instance[1], instance[2])

    for plane in planes:
        _in = 0
        _out = 0
        normal = plane.xyz
        w = plane.w
        if glm.dot(normal, pos + glm.vec3(0.5, 0.5, 0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(-0.5, 0.5, 0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(0.5, -0.5, 0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(0.5, 0.5, -0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(-0.5, -0.5, 0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(0.5, -0.5, -0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(-0.5, 0.5, -0.5)) + w < 0:
            _out += 1
        else:
            _in += 1
        if glm.dot(normal, pos + glm.vec3(-0.5, -0.5, -0.5)) + w < 0:
            _out += 1
        else:
            _in += 1

        
        if not _in:
            return False
    return True




def update_position(delta_time):
    multiplier = 7 * delta_time

    camera_position[1] += user_input[1] * multiplier

    if user_input[0] or user_input[2]:
        angle = camera_rotation[0] - math.atan2(user_input[2], user_input[0]) + math.tau / 4

        camera_position[0] += math.cos(angle) * multiplier
        camera_position[2] += math.sin(angle) * multiplier


@window.event
def on_mouse_motion(x, y, delta_x, delta_y):
    if mouse_captured[0]:

        camera_rotation[0] += delta_x * 0.004
        camera_rotation[1] += delta_y * 0.004

        camera_rotation[1] = max(-math.tau / 4, min(math.tau / 4, camera_rotation[1]))

@window.event
def on_mouse_drag(x, y, delta_x, delta_y, buttons, modifiers):
    on_mouse_motion(x, y, delta_x, delta_y)
	
@window.event
def on_key_press(key, modifiers):
    if not mouse_captured[0]:
        return

    if   key == pyglet.window.key.D: user_input[0] += 1
    elif key == pyglet.window.key.A: user_input[0] -= 1
    elif key == pyglet.window.key.W: user_input[2] += 1
    elif key == pyglet.window.key.S: user_input[2] -= 1

    elif key == pyglet.window.key.SPACE : user_input[1] += 1
    elif key == pyglet.window.key.LSHIFT: user_input[1] -= 1

@window.event
def on_key_release(key, modifiers):
    if not mouse_captured[0]:
        return

    if   key == pyglet.window.key.D: user_input[0] -= 1
    elif key == pyglet.window.key.A: user_input[0] += 1
    elif key == pyglet.window.key.W: user_input[2] -= 1
    elif key == pyglet.window.key.S: user_input[2] += 1

    elif key == pyglet.window.key.SPACE : user_input[1] -= 1
    elif key == pyglet.window.key.LSHIFT: user_input[1] += 1

@window.event
def on_mouse_press(x, y, button, modifiers):
    mouse_captured[0] = not mouse_captured[0]
    window.set_exclusive_mouse(mouse_captured[0])

    return

t = [0]

def tick(delta_time):
    t[0] += delta_time

pyglet.clock.schedule(tick)

def init_all():

    glCreateVertexArrays(1, vao)


    glCreateBuffers(1, vbo)
    glNamedBufferStorage(
        vbo, 
        ctypes.sizeof(GLfloat) * len(vertices), 
        (GLfloat * len(vertices))(*vertices), 
        0
    )

    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 9 * ctypes.sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 0)
    glVertexArrayAttribBinding(vao, 0, 0)
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0)

    glEnableVertexArrayAttrib(vao, 1)
    glVertexArrayAttribBinding(vao, 1, 0)
    glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat))

    glEnableVertexArrayAttrib(vao, 2)
    glVertexArrayAttribBinding(vao, 2, 0)
    glVertexArrayAttribFormat(vao, 2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat))
    
    instances = tuple(item for sublist in instanced_array for item in sublist)
    glCreateBuffers(1, instance_vbo)
    glNamedBufferStorage(
        instance_vbo, 
        ctypes.sizeof(GLfloat) * len(instances), 
        (GLfloat * len(instances))(*instances), 
        0
    )

    glVertexArrayVertexBuffer(vao, 1, instance_vbo, 0, 4 * sizeof(GLfloat))
    glVertexArrayBindingDivisor(vao, 1, 1)

    glVertexArrayAttribBinding(vao, 3, 1)
    glEnableVertexArrayAttrib(vao, 3)
    glVertexArrayAttribFormat(vao, 3, 3, GL_FLOAT, GL_FALSE, 0)

    glVertexArrayAttribBinding(vao, 4, 1)
    glEnableVertexArrayAttrib(vao, 4)
    glVertexArrayAttribFormat(vao, 4, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat))


    glCreateBuffers(1, ibo)
    glNamedBufferStorage(ibo, 
        ctypes.sizeof(GLuint) * len(indices), 
        (GLuint * len(indices))(*indices),
        0
    )

    glVertexArrayElementBuffer(vao, ibo)


    glCreateBuffers(1, icbo)
    glNamedBufferStorage(
        icbo,
        ctypes.sizeof(GLuint) * len(cmd),
        (GLuint * len(cmd))(*cmd),
        0
    )

    glCreateBuffers(1, mat_ubo)
    glNamedBufferStorage(
        mat_ubo, 
        19 * ctypes.sizeof(GLfloat),
        None,
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT 
    )
    mapped_ubo[0] = glMapNamedBufferRange(
        mat_ubo, 
        0, 
        19 * ctypes.sizeof(GLfloat), 
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT
    )
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, mat_ubo)
    
    light_proj = glm.ortho(-64, 64, -64, 64, 0.1, 1024) 
    light_view = glm.lookAt(glm.vec3(light_instance[0], light_instance[1], light_instance[2]), 
                                  glm.vec3(0.0), 
                                  glm.vec3(0.0, 1.0, 0.0)); 

    light_proj_view = light_proj * light_view; 

    glProgramUniformMatrix4fv(shadow_program, 2, 1, GL_FALSE, glm.value_ptr(light_proj_view))
    glProgramUniformMatrix4fv(program, 2, 1, GL_FALSE, glm.value_ptr(light_proj_view))
    light_data = [
        light_instance[0], light_instance[1], light_instance[2], # Light Pos
        0.05, # Ambient
        0.75, # Specular
        1.0, 0.0, 0.0
    ]
    glCreateBuffers(1, light_ubo)
    glNamedBufferStorage(
        light_ubo,
        (len(light_data)) * ctypes.sizeof(GLfloat),
        (GLfloat * len(light_data))(*light_data),
        0
    )
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, light_ubo)


    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, tao)
    glTextureParameteri(tao, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTextureParameteri(tao, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTextureParameteri(tao, GL_TEXTURE_MAX_ANISOTROPY, 8)
    glTextureStorage3D(
        tao, 4, GL_RGBA8, 16, 16, 256
    )
    stone_image = pyglet.image.load(f"textures/stone.png").get_image_data()
    data = stone_image.get_data("RGBA", stone_image.width * 4)
    glTextureSubImage3D(
        tao, 0, 0, 0, 0,
        stone_image.width, stone_image.height, 1, 
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    cobblestone_image = pyglet.image.load(f"textures/cobblestone.png").get_image_data()
    data = cobblestone_image.get_data("RGBA", cobblestone_image.width * 4)
    glTextureSubImage3D(
        tao, 0, 0, 0, 1,
        cobblestone_image.width, cobblestone_image.height, 1, 
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    light_image = pyglet.image.load(f"textures/light_block.png").get_image_data()
    data = light_image.get_data("RGBA", light_image.width * 4)
    glTextureSubImage3D(
        tao, 0, 0, 0, 2,
        light_image.width, light_image.height, 1, 
        GL_RGBA, GL_UNSIGNED_BYTE, data
    )
    glGenerateTextureMipmap(tao)
    
    glBindTextureUnit(0, tao)
    glProgramUniform1i(program, 0, 0)

    glCreateFramebuffers(1, shadow_fbo)

    glCreateTextures(GL_TEXTURE_2D, 1, shadow_map)
    glTextureStorage2D(shadow_map, 1, GL_DEPTH_COMPONENT16, 2048, 2048)

    glTextureParameteri(shadow_map, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTextureParameteri(shadow_map, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTextureParameteri(shadow_map, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(shadow_map, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTextureParameteri(shadow_map, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER) 
    glTextureParameteri(shadow_map, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    
    clamp_color = glm.vec4(1.0)
    glTextureParameterfv(shadow_map, GL_TEXTURE_BORDER_COLOR, glm.value_ptr(clamp_color))

    glNamedFramebufferTexture(shadow_fbo, GL_DEPTH_ATTACHMENT, shadow_map, 0)
    glNamedFramebufferDrawBuffer(shadow_fbo, GL_NONE)
    glNamedFramebufferReadBuffer(shadow_fbo, GL_NONE)

    assert glCheckNamedFramebufferStatus(shadow_fbo, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
    
    # glBindTextureUnit(1, shadow_map)
    glBindTextureUnit(2, shadow_map)
    glProgramUniform1i(program, 1, 2)

    glCreateVertexArrays(1, quad_vao)
    
    glCreateBuffers(1, quad_vbo)
    glNamedBufferStorage(quad_vbo, 
        ctypes.sizeof(GLfloat) * len(quad), 
        (GLfloat * len(quad))(*quad), 
        0
    )

    glVertexArrayVertexBuffer(quad_vao, 0, quad_vbo, 0, 4 * sizeof(GLfloat))

    glEnableVertexArrayAttrib(quad_vao, 0)
    glVertexArrayAttribBinding(quad_vao, 0, 0)
    glVertexArrayAttribFormat(quad_vao, 0, 2, GL_FLOAT, GL_FALSE, 0)

    glEnableVertexArrayAttrib(quad_vao, 1)
    glVertexArrayAttribBinding(quad_vao, 1, 0)
    glVertexArrayAttribFormat(quad_vao, 1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat))


def create_fbo():
    glCreateFramebuffers(1, msaa_fbo)

    glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, msaa_cbo)
    glTextureStorage2DMultisample(msaa_cbo, 8, GL_RGB16F, dim[0], dim[1], GL_TRUE)

    glNamedFramebufferTexture(msaa_fbo, GL_COLOR_ATTACHMENT0, msaa_cbo, 0)

    glCreateRenderbuffers(1, msaa_dbo)
    glNamedRenderbufferStorageMultisample(msaa_dbo, 8, GL_DEPTH24_STENCIL8, dim[0], dim[1])

    glNamedFramebufferRenderbuffer(msaa_fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msaa_dbo)

    assert glCheckNamedFramebufferStatus(msaa_fbo, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
    
    glCreateFramebuffers(1, pp_fbo)

    glCreateTextures(GL_TEXTURE_2D, 1, pp_cbo)
    glTextureStorage2D(pp_cbo, 1, GL_RGB16F, dim[0], dim[1])
    glTextureParameteri(pp_cbo, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTextureParameteri(pp_cbo, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glNamedFramebufferTexture(pp_fbo, GL_COLOR_ATTACHMENT0, pp_cbo, 0)

    glCreateRenderbuffers(1, pp_dbo)
    glNamedRenderbufferStorage(pp_dbo, GL_DEPTH24_STENCIL8, dim[0], dim[1])

    glNamedFramebufferRenderbuffer(pp_fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pp_dbo)

    assert glCheckNamedFramebufferStatus(pp_fbo, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

    glBindTextureUnit(1, pp_cbo)
    glProgramUniform1i(pprcs_program, 0, 1)

@window.event
def on_resize(width, height):
    glDeleteFramebuffers(1, msaa_fbo)
    glDeleteTextures(1, msaa_cbo)
    glDeleteRenderbuffers(1, msaa_dbo)
    glDeleteFramebuffers(1, pp_fbo)
    glDeleteTextures(1, pp_cbo)
    glDeleteRenderbuffers(1, pp_dbo)

    dim[0] = width
    dim[1] = height

    create_fbo()

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, height)
    
def update(delta_time):
    if not mouse_captured[0]:
        user_input[0] = 0
        user_input[1] = 0
        user_input[2] = 0

    update_position(delta_time)

def update_matrices():
    p_matrix = glm.perspective(
                glm.radians(90),
                dim[0] / dim[1], 0.1, 500)

    v_matrix = glm.mat4(1.0)
    v_matrix = glm.rotate(v_matrix, camera_rotation[1], -glm.vec3(1.0, 0.0, 0.0))
    v_matrix = glm.rotate(v_matrix, -camera_rotation[0] - math.tau / 4, -glm.vec3(0.0, 1.0, 0.0))
    v_matrix = glm.translate(v_matrix, -camera_position)
    

    vp_matrix = p_matrix * v_matrix

    ctypes.memmove(mapped_ubo[0], glm.value_ptr(vp_matrix), 16 * ctypes.sizeof(GLfloat))
    ctypes.memmove(mapped_ubo[0] + 16 * ctypes.sizeof(GLfloat), glm.value_ptr(camera_position), 3 * ctypes.sizeof(GLfloat))
    glFlushMappedNamedBufferRange(mat_ubo, 0, 19 * ctypes.sizeof(GLfloat))
"""
    mat = glm.transpose(vp_matrix)
    for i in range(4): 
            Frustum.left[i]      = mat[3][i] + mat[0][i]
            Frustum.right[i]     = mat[3][i] - mat[0][i]
            Frustum.bottom[i]    = mat[3][i] + mat[1][i]
            Frustum.top[i]       = mat[3][i] - mat[1][i]
            Frustum.near[i]      = mat[3][i] + mat[2][i]
            Frustum.far[i]       = mat[3][i] - mat[2][i]
            
    Frustum.left /= glm.length(Frustum.left.xyz)
    Frustum.right /= glm.length(Frustum.right.xyz)
    Frustum.bottom /= glm.length(Frustum.bottom.xyz)
    Frustum.top /= glm.length(Frustum.top.xyz)
    Frustum.near /= glm.length(Frustum.near.xyz)
    Frustum.far /= glm.length(Frustum.far.xyz)

    results = map(check_in_frustum, instanced_array)
    cmd = []
    contiguous = False
    for i, results in enumerate(results):
        if results:
            if contiguous:
                cmd[-4] += 1
            else:
                cmd.extend((len(indices), 1, 0, 0, i))
                contiguous = True
        else:
            contiguous = False
    glNamedBufferSubData(icbo, 0, ctypes.sizeof(GLuint) * len(cmd), (GLuint * len(cmd))(*cmd))
"""

pyglet.clock.schedule(update)
window.set_exclusive_mouse(True)


@window.event
def on_draw():
    update_matrices()

    while len(fences) > 3:
        fence = fences.popleft()
        glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 2147483647)
        glDeleteSync(fence)

    glProgramUniform1f(pprcs_program, 1, 1)

    glViewport(0, 0, 2048, 2048)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
    glDepthMask(GL_TRUE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)
    glUseProgram(shadow_program)
    glBindVertexArray(vao)
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, icbo)

    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, None, len(cmd) // 5, 0)

    glViewport(0, 0, dim[0], dim[1])
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)    
    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(program)
    glBindVertexArray(vao)
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, icbo)
    
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, None, len(cmd) // 5, 0)

    glBlitNamedFramebuffer(
        msaa_fbo, pp_fbo, 0, 0, dim[0], dim[1], 0, 0, dim[0], dim[1], GL_COLOR_BUFFER_BIT, GL_LINEAR
    )

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(pprcs_program)
    glBindVertexArray(quad_vao)

    glDrawArrays(GL_TRIANGLES, 0, 6)

    glUseProgram(0)
    glBindVertexArray(0)

    fps_display.draw()

    fences.append(glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0))

def main():
    init_all()
    create_fbo()
    pyglet.app.run()

if __name__ == "__main__":
    main()
