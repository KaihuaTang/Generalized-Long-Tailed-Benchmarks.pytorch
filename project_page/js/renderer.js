import * as THREE from 'three';
import { OrbitControls } from './jsm/controls/OrbitControls.js';
import { RGBELoader } from './jsm/loaders/RGBELoader.js';


let camera_dynamic, scene_dynamic, renderer_dynamic, container_dynamic;
let material;
let cubeRenderTarget, cubeCamera;
let cube, sphere, torus, controls;
let name_path;

init();

function init() {
    document.getElementById("myModal").style.display = 'block';

    // Get the modal
    var modal = document.getElementById("myModal");
    // Get the button that opens the modal
    var btn_list = document.getElementsByClassName("named_button");
    for (let i = 0; i < btn_list.length; i++) {
        // btn_list[i].onclick = on_click_func(btn_list[i], modal)
        btn_list[i].onclick = function() {
            modal.style.display = "block";
            console.log(btn_list[i].value);
            name_path = 'full_[' + btn_list[i].value + ']';
            new RGBELoader()
            .setPath( 'asset/' )
            .load( name_path + '.hdr', function ( texture ) {
                texture.mapping = THREE.EquirectangularReflectionMapping;
                scene_dynamic.background = texture;
                scene_dynamic.environment = texture;
            } );
            rendering_fn();
        };
    }
    
    // Get the <span> element that closes the modal
    
    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
        renderer_dynamic.setAnimationLoop(null);
      }
    }

    dynamic_init()

    window.addEventListener( 'resize', onWindowResize() );
    document.getElementById("myModal").style.display = 'none';
}

function dynamic_init() {
    container_dynamic = document.getElementById("threejs-dynamic-container");
    camera_dynamic = new THREE.PerspectiveCamera(60, container_dynamic.clientWidth / container_dynamic.clientHeight, 1, 1000);
    camera_dynamic.position.z = 75;
    scene_dynamic = new THREE.Scene();
    scene_dynamic.rotation.y = 0.5;

    renderer_dynamic = new THREE.WebGLRenderer( { antialias: true } );
    renderer_dynamic.setPixelRatio( 2 );
    // renderer_dynamic.setSize( container_dynamic.clientWidth, container_dynamic.clientHeight );
    // renderer_dynamic.domElement.setAttribute('style', 'width: 32%')
    renderer_dynamic.outputEncoding = THREE.sRGBEncoding;
    renderer_dynamic.toneMapping = 1;
    cubeRenderTarget = new THREE.WebGLCubeRenderTarget( 1024 );
    cubeRenderTarget.texture.type = THREE.HalfFloatType;

    cubeCamera = new THREE.CubeCamera( 1, 1000, cubeRenderTarget );

    material = new THREE.MeshStandardMaterial( {
        envMap: cubeRenderTarget.texture,
        roughness: 0.05,
        metalness: 1
    } );
    sphere = new THREE.Mesh( new THREE.IcosahedronGeometry( 15, 8 ), material );
    scene_dynamic.add( sphere );
    const material2 = new THREE.MeshStandardMaterial( {
        roughness: 0.1,
        metalness: 0
    } );

    cube = new THREE.Mesh( new THREE.BoxGeometry( 15, 15, 15 ), material2 );
    scene_dynamic.add( cube );
    torus = new THREE.Mesh( new THREE.TorusKnotGeometry( 8, 3, 128, 16 ), material2 );
    scene_dynamic.add( torus );

    container_dynamic.appendChild( renderer_dynamic.domElement );

    controls = new OrbitControls( camera_dynamic, renderer_dynamic.domElement );
    controls.autoRotate = true;

}

function animation( msTime ) {

    const time = msTime / 1000;

    cube.position.x = Math.cos( time ) * 30;
    cube.position.y = Math.sin( time ) * 30;
    cube.position.z = Math.sin( time ) * 30;

    cube.rotation.x += 0.02;
    cube.rotation.y += 0.03;

    torus.position.x = Math.cos( time + 10 ) * 30;
    torus.position.y = Math.sin( time + 10 ) * 30;
    torus.position.z = Math.sin( time + 10 ) * 30;

    torus.rotation.x += 0.02;
    torus.rotation.y += 0.03;

    cubeCamera.update( renderer_dynamic, scene_dynamic );

    controls.update();

    renderer_dynamic.render( scene_dynamic, camera_dynamic );
}

function onWindowResize() {
    camera_dynamic.aspect = container_dynamic.clientWidth / container_dynamic.clientHeight;
    camera_dynamic.updateProjectionMatrix();

    renderer_dynamic.setSize( container_dynamic.clientWidth, container_dynamic.clientHeight );
}

function rendering_fn() {
    var img = document.getElementById("pano-img");
    img.src = "img/pano/" + name_path + '.png';
    img.setAttribute('style', 'width:100%');
    renderer_dynamic.setAnimationLoop( animation );
}