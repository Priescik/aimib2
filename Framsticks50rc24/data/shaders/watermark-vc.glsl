glcompat_attrs=6 //texture+vertcolors

[VERTEX]

#include "common.glsl"

attribute vec4 a_vert;
attribute vec4 a_texcoord;
attribute vec4 a_color;
varying mediump vec4 v_color;
varying mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    gl_Position = u_projmodmat * a_vert;
    v_texcoord=vec2(u_texmat*vec4(a_texcoord));
    v_texcoord2=vec2(u_texmat2*a_vert);
    v_color=a_color/255.0;
}

[FRAGMENT]

#include "common.glsl"

varying mediump vec4 v_color;
varying mediump vec2 v_texcoord,v_texcoord2;

void main()
{
    mediump vec4 c=v_color;
    c*=texture2D(u_tex,v_texcoord)*texture2D(u_tex2,v_texcoord2);
    gl_FragColor=c;
}
