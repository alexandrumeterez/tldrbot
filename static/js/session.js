$(document).ready(function() {
    $("#btnSend").click(function(){

        if($('#questionForm').val() === '' || $('#docForm').val() === ''){
        }
        else{
            var $message_html = "";
            $message_html += "<div class=\"row\">"
            $message_html += "<div class=\"message-user col-sm offset-sm-7\">";
            $message_html += $('#questionForm').val();
            $message_html += "</div></div>";
            $("#chatbox").append($message_html);
            $('#questionForm').val('');
            $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight);
        }
    });

    $('#questionForm').keypress(function (e) {
     var key = e.which;
     if(key == 13)  // the enter key code
      {
        return false;
      }
    });
});