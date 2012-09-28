// requires:
//     jquery.js

'use strict';

//////// utilities //////// 
String.prototype.format = function() {
  var args = arguments;
  return this.replace(/{(\d+)}/g, function(match, number) { 
    return typeof args[number] != 'undefined'
      ? args[number]
      : match
    ;
  });
};

//////// webffoct ////////

var webffoct = new function() {
	var self = this; // for private functions

	this.API_ROOT = '/api/';

	this.url = function() {
		var argsarr = [];
		for (var i = 0; i < arguments.length; i ++) {
			argsarr.push(arguments[i]);
		}
		return this.API_ROOT + argsarr.join('/');
	};

	this.outerHtml = function(elem) {
		return $(elem).clone().wrap('<p>').parent().html();
	}

	this.getMasters = function(args) {
		// args: {
		//   success:
		//   error: 
		// }
		$.ajax({
			url: this.url('masters/'),
			dataType: 'json',
			type: 'GET',
			success: args.success,
			error: args.error
		});
	};

}; // var webffoct

