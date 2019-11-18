var scene, camera, renderer, light;
var earthRotY = 0, moonRotY = 0;
var radY = 0, radZ = -0.3;
var moonDist = 70;
var earthRadius = 25;
var earthMesh, tmpMesh;
var moonMesh;
var positionHistory = [];
var lastPos, diffMove, lastEarthScale;
var ping = 0;


function init(width, height) {
    scene = new THREE.Scene();
    // Setup cameta with 45 deg field of view and same aspect ratio
    var aspect = width / height;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    // Set the camera to 400 units along `z` axis
    camera.position.set(0, 0, 400);

    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer:true, alpha: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.domElement.classList.add("earth");
    document.body.appendChild(renderer.domElement);
}

function initLight() {
    light = new THREE.SpotLight(0xffffff);
    // Position the light slightly to a side to make 
    // shadows look better.
    light.position.set(400, 100, 1000);
    light.castShadow = true;
    scene.add(light);
}

function initEarth() {
    // Load Earth texture and create material from it
    var earthTexture = THREE.ImageUtils.loadTexture(earthBase64);
    earthTexture.minFilter = THREE.NearestFilter;
    var earthMaterial = new THREE.MeshLambertMaterial({
        map: earthTexture,
    });
    // Create a sphere 25 units in radius and 16 segments
    // both horizontally and vertically.
    var earthGeometry = new THREE.SphereGeometry(earthRadius, 16, 16);
    earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    earthMesh.receiveShadow = true;
    earthMesh.castShadow = true;
    // Add Earth to the scene
    scene.add(earthMesh);
}

function initMoon() {
    var moonTexture = THREE.ImageUtils.loadTexture(moonBase64);
    moonTexture.minFilter = THREE.NearestFilter;
    var moonMaterial = new THREE.MeshLambertMaterial({
        map: moonTexture,
    });
    var moonGeometry =  new THREE.SphereGeometry(earthRadius * 0.273, 10, 10);
    moonMesh = new THREE.Mesh(moonGeometry, moonMaterial);
    moonMesh.receiveShadow = true;
    moonMesh.castShadow = true;
    scene.add(moonMesh);
}

function initPlane() {
    // The plane needs to be large to be sure it'll always intersect
    var tmpGeometry = new THREE.PlaneGeometry(1000, 1000, 1, 1);
    tmpGeometry.position = new THREE.Vector3(0, 0, 0);
    tmpMesh = new THREE.Mesh(tmpGeometry);
}

var intersects = [];

function checkIntersect(vector) {
    // Unproject camera distortion (fov, aspect ratio)
    vector.unproject(camera);
    var norm = vector.sub(camera.position).normalize();
    var ray = new THREE.Raycaster(camera.position, norm);
    // Cast a line from our camera to the tmpMesh and see where these
    // two intersect. That's our 2D position in 3D coordinates.
    intersects = ray.intersectObject(tmpMesh);

    return intersects[0].point;
}

// Update position of objects in the scene
function update() {
    if (positionHistory.length === 0) {
       return;
    }

    earthRotY += 0.007;

    ping++;
    if (ping < 10) {
        lastPos[0] += diffMove[0];
        lastPos[1] += diffMove[1];
        lastPos[2] += diffMove[2];
    }

    //dsadsabjdkd
    //var vector = new THREE.Vector3(0.1, 0.1, 0.5);
    var vector = new THREE.Vector3(lastPos[0], lastPos[1], 0.5);
    var intersect = checkIntersect(vector);

    earthMesh.rotation.y = earthRotY;

    // With position from OpenCV I could possibly move the Earth outside of the window
    if (intersects.length === 1) {
        var point = intersects[0].point;
        earthMesh.position.x = point.x;
        earthMesh.position.y = point.y;


        // X pos + radius
        var vector = new THREE.Vector3(lastPos[0] + lastPos[2], lastPos[1], 0.5);
        var intersect = checkIntersect(vector);

        var newEarthRadius = Math.abs(intersect.x - earthMesh.position.x);
        var earthScale = newEarthRadius / earthRadius;

        earthMesh.scale.set(earthScale, earthScale, earthScale);
        moonMesh.scale.set(earthScale, earthScale, earthScale);

        lastEarthScale = earthScale;
    }

    moonRotY += 0.005;
    radY += 0.03;
    radZ += 0.0005;

    // Update Moon position
    x = lastEarthScale * moonDist * Math.cos(radZ) * Math.sin(radY);
    y = lastEarthScale * moonDist * Math.sin(radZ) * Math.sin(radY);
    z = lastEarthScale * moonDist * Math.cos(radY);

    moonMesh.position.set(x + earthMesh.position.x, y + earthMesh.position.y, z);
    moonMesh.rotation.y = moonRotY;
}

function render() {
    update();
    
    renderer.setClearColor(0xffffff, 0);
    renderer.render(scene, camera);
    // Schedule another frame
    requestAnimationFrame(render);
}


document.addEventListener('DOMContentLoaded', function(event) {

    // var video = document.querySelector("#video");
    // var constraints = { video: { facingMode: "user", minWidth: window.innerWidth }, audio: false } ;

    // function cameraStart() {

    //     navigator.mediaDevices
    //         .getUserMedia(constraints)
    //         .then(function(stream) {
            
    //             track = stream.getTracks()[0];
    //             video.srcObject = stream;
            // video.oncanplay = function() {
                //console.log(video.clientWidth, video.clientHeight);
                //init(video.clientWidth, video.clientHeight);
                init(window.innerWidth, window.innerHeight);
                initEarth();
                initMoon();
                initLight();
                initPlane();

                requestAnimationFrame(render);
                    

            // }, function() {}

        // })
// }
})
// window.addEventListener("load", cameraStart, false);

var ws = new WebSocket('ws://localhost:9000');
ws.onopen = function() {
    console.log('onopen');
};
ws.onmessage = function (event) {
    data = JSON.parse(event.data.split("'").join('"'));
    console.log("Data: ", data);

    let msg = data;

    positionHistory.push({
        x:  msg.x * 2 - 1,
        y: - msg.y * 2 + 1,
        // w: msg.w * 2 - 1,
        // h: - msg.h * 2 + 1
    });

    if (positionHistory.length > 10) {
        positionHistory.shift();
    }

    var xCoords = [], yCoords = [], radiuses = [];
    for (var i = math.max(positionHistory.length - 2, 0); i < positionHistory.length; i++) {
        xCoords.push(positionHistory[i].x);
        yCoords.push(positionHistory[i].y);
    }
    
    for (var i = 0; i < positionHistory.length; i++) {
        radiuses.push(positionHistory[i].radius);
    }

    var posX = math.mean(xCoords);
    var posY = math.mean(yCoords);
    var radius = math.mean(radiuses);

    var targetPos = [posX, posY, radius];
    if (!lastPos) {
        lastPos = targetPos;
    }
    diffMove = [(targetPos[0] - lastPos[0]) / 4, (targetPos[1] - lastPos[1]) / 4, (targetPos[2] - lastPos[2]) / 4]

    ping = 0;
}